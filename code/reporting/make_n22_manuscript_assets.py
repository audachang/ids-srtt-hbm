#!/usr/bin/env python3
"""Format validated saved estimates into the T9 manuscript assets; never fit models.

Run from any directory with envs/nilearn-0131/bin/python. Inputs are checked against
saved task validation hashes before reading. Outputs are confined to n22_* assets.
LOO extrema select existing rows; coverage counts are saved eligible n values.
"""
from pathlib import Path
import hashlib
import json
import math
import os
import shutil
import re
import subprocess

os.environ.setdefault('MPLCONFIGDIR', '/tmp/t9-t9-tables/mpl')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Rectangle
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
GLM = ROOT / 'results/glm_volume/reanalysis_20261018_updated_n22'
BEH = ROOT / 'results/behavioral/reanalysis_20261018_updated_n22'
TABLES = ROOT / 'manuscript/tables'
FIGURES = ROOT / 'manuscript/figures'
SOURCES, OUTPUTS = {}, {}
FIGURE_SOURCE_ROWS = {}
FACT_PATH = ROOT / 'docs/t9/facts_n22.json'
FACTS = json.loads(FACT_PATH.read_text())
FACT_VALUES = {f['id']:f['value'] for f in FACTS}
FACT_INDEX, FACT_KEYS, USED_IDS, TABLE_IDS, FIGURE_IDS = {}, {}, set(), {}, {}


def canonical(v):
    return format(float(v), '.15g') if isinstance(v, (int, float)) else str(v)


for fact in FACTS:
    source, column, keys = fact['source'], fact['column'], fact['row_key']
    if not isinstance(column, str) or 'json_pointer' in keys or 'line' in keys:
        continue
    names = tuple(sorted(keys))
    FACT_KEYS.setdefault(source, set()).add(names)
    identity = tuple((k, canonical(keys[k])) for k in names)
    FACT_INDEX.setdefault((source, identity, column), []).append(fact['id'])


def trace(row, columns):
    source = row.attrs.get('source')
    for names in FACT_KEYS.get(source, []):
        if not all(k in row for k in names):
            continue
        identity = tuple((k, canonical(row[k])) for k in names)
        for column in columns:
            ids = FACT_INDEX.get((source, identity, column), [])
            for fact_id in ids:
                expected, actual = FACT_VALUES[fact_id], row[column]
                if expected is None:
                    if not pd.isna(actual):raise ValueError(f'Ledger mismatch: {fact_id}')
                elif isinstance(expected,(float,int)):
                    if not math.isclose(float(actual),float(expected),rel_tol=1e-12,abs_tol=1e-14):
                        raise ValueError(f'Ledger mismatch: {fact_id}')
            USED_IDS.update(ids)


def qcell(row, column='q'):
    trace(row, [column, 'n'])
    return r'\textemdash' if 'n' in row and row['n'] < 2 else q(row[column])

ROIS = ['LeftSMA', 'RightSMA', 'RightPallidum', 'RightThalamusProper']
FOCUS = ROIS + ['RightAccumbensArea']
MODELS = ['reference', 'censored', 'spm', 'spm_derivative', 'motion12_acompcor8',
          'precision_weighted', 'errors_omissions', 'rt_modulator', 'rt_shared_modulator']
MODEL_NAMES = {'reference':'Uncensored reference', 'censored':'Volume censoring',
    'spm':'SPM HRF', 'spm_derivative':'SPM HRF + derivative',
    'motion12_acompcor8':'12 motion + aCompCor', 'precision_weighted':'Precision weighting',
    'errors_omissions':'Errors/omissions adjustment', 'rt_modulator':'Condition-specific RT adjustment',
    'rt_shared_modulator':'Shared RT adjustment'}
LABELS = {'initial':'Initial Str1--random', 'early':'Early DiD', 'middle':'Middle DiD',
    'late':'Late DiD', 'pooled_r36':'Pooled runs 3--6', 'early_r34':'Runs 3--4',
    'middle_r56':'Runs 5--6', 'Str2_minus_Str1':'Str2--Str1 trend',
    'Str2_minus_Ran':'Str2--random trend', 'early_minus_middle':'Early--middle',
    'early_minus_late':'Early--late', 'middle_minus_late':'Middle--late'}
plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':10, 'pdf.fonttype':42,
                     'savefig.dpi':180, 'axes.spines.top':False, 'axes.spines.right':False})
PDF_META = {'Creator':'T9 validated manuscript asset formatter', 'CreationDate':None,
            'ModDate':None}


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def checked(p):
    p = Path(p)
    v = p.parent / 'validation.json'
    j = json.loads(v.read_text())
    if not j.get('passed'):
        raise ValueError(f'Validation has not passed: {v}')
    expected = j['output_sha256'][p.name]
    actual = sha(p)
    if actual != expected:
        raise ValueError(f'Source SHA-256 mismatch: {p}')
    SOURCES[str(p.relative_to(ROOT))] = {'sha256':actual,
        'validation_path':str(v.relative_to(ROOT)), 'validation_sha256':sha(v)}
    return p


def read(p):
    d = pd.read_csv(checked(p))
    d.attrs['source'] = str(p.relative_to(ROOT))
    return d


def register(p):
    assert p.parent in (TABLES, FIGURES) and p.name.startswith('n22_')
    OUTPUTS[str(p.relative_to(ROOT))] = sha(p)


def tex(s):
    return str(s).replace('&', r'\&').replace('%', r'\%').replace('_', r'\_')


def roi(s):
    if s.endswith('SMA'):
        return s.replace('SMA',' BA6')
    s = s.replace('ThalamusProper','Thalamus').replace('AccumbensArea','Accumbens')
    s = s.replace('DLPFC46','BA46').replace('DLPFC9','BA9')
    return re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', s)


def f(x, digits=3, signed=False):
    if pd.isna(x) or not math.isfinite(float(x)):
        return r'\textemdash'
    s = f'{float(x):+.{digits}f}' if signed else f'{float(x):.{digits}f}'
    if float(s) == 0:
        s = f'{0:.{digits}f}'
    return '$' + s + '$'


def q(x):
    if pd.isna(x):
        return r'\textemdash'
    return '$<.001$' if x < .001 else '$' + f'{x:.3f}'.removeprefix('0') + '$'


def count(x):
    return f'{int(x):,}'


def ci(r, mean='mean', lo='ci_low', hi='ci_high', signed=True):
    trace(r, [mean, lo, hi, 'n'])
    if ('n' in r and r['n'] < 2) or any(pd.isna(r[c]) for c in [mean,lo,hi]):
        return r'\textemdash'
    return f(r[mean],signed=signed) + ' [' + f(r[lo],signed=signed) + ', ' + f(r[hi],signed=signed) + ']'


def sub(d, **kw):
    for k,v in kw.items():
        d = d[d[k].isin(v) if isinstance(v,list) else d[k].eq(v)]
    return d


def one(d, **kw):
    a=sub(d,**kw)
    if len(a)!=1: raise ValueError(f'Expected one saved row: {kw}; got {len(a)}')
    return a.iloc[0]


def table(name, caption, headers, rows, widths=None, notes=''):
    """Longtables work in both existing manuscript preambles without extra packages."""
    p=TABLES/(name+'.tex')
    widths=widths or ['l']+['r']*(len(headers)-1)
    header=' & '.join(headers)+r' \\'
    body=['% Generated by code/reporting/make_n22_manuscript_assets.py; do not edit.',
        r'\begingroup\def\baselinestretch{1}\footnotesize',r'\setlength{\tabcolsep}{3pt}',
        r'\renewcommand{\arraystretch}{1.12}',r'\begin{longtable}{'+''.join(widths)+'}',
        r'\caption{'+caption+r'}\label{tab:'+name.replace('_','-')+r'}\\',
        r'\toprule',header,r'\midrule\endfirsthead',
        r'\multicolumn{'+str(len(headers))+r'}{l}{\tablename\ \thetable\ (continued)}\\',
        r'\toprule',header,r'\midrule\endhead',
        r'\midrule\multicolumn{'+str(len(headers))+r'}{r}{Continued on next page}\\\endfoot',
        r'\bottomrule\endlastfoot']
    body += [' & '.join(map(str,row))+r' \\' for row in rows]
    body += [r'\end{longtable}']
    if notes: body += [r'\noindent\parbox{\linewidth}{\footnotesize '+notes+r'}\par']
    body += [r'\endgroup','']
    TABLE_IDS[str(p.relative_to(ROOT))] = sorted(USED_IDS)
    USED_IDS.clear()
    p.write_text('\n'.join(body));register(p)


ROI_CAP = (r'$N=22$, uncensored reference model, at least 80\% observed ROI support. '
           r'Signed estimates and pointwise 95\% Student-$t$ CIs; actual eligible $n$. ')
ROI_NOTE = (r'BA6 denotes the Brodmann parcel. Initial and DiD estimates use two-run sums; '
    r'within-run estimates use equal-run means. Positive within-run estimates mean Str2 $>$ Str1. '
    r'Positive DiD denotes a relative contrast shift, not necessarily a structured-response increase. '
    r'CIs are unadjusted for multiplicity and selection. Untestable rows are shown as '
    r'\textemdash\ with $n$; their planned BH probability slots are retained as $p=1$.')
BEH_NOTE = (r'Approximate tests using participant df = 21 ($N=22$) and 19 ($N=20$). '
    r'The samples overlap and comparisons are descriptive. RT CIs are pointwise $t$ intervals; '
    r'error CIs are pointwise participant-bootstrap percentile intervals. '
    r'Phase changes and phase pairs are ratios of ratios for RT and ratios of odds ratios for binomial errors. '
    r'BH covers 24 contrasts separately by cohort, response policy, and outcome/model; '
    r'intervals are not simultaneous or inversions of the bootstrap-based error tests. '
    r'Imaging coverage rules do not apply to behavioral models.')


def contrast(c):
    if c in LABELS:return LABELS[c]
    if c.endswith('_adjusted_change'):return c.split('_')[0].title()+' adjusted change'
    if re.fullmatch(r'run[3-6]', c): return 'Run '+c[3:]
    if c.startswith('run'):
        a,b=c.split('_',1)
        return 'Run '+a[3:]+': '+b.replace('_minus_', '--').replace('Ran','random')
    return c.replace('_minus_', '--').replace('_',' ')


def neural_tables():
    stages=read(GLM/'roi_reporting/stage_results.csv')
    within=read(GLM/'roi_reporting/within_run_results.csv')
    trends=read(GLM/'roi_reporting/trend_results.csv')
    direct=read(GLM/'roi_reporting/direct_phase_results.csv')
    primary=lambda d:sub(d,cohort='N22',model='reference',policy='observed80')
    rows=[]
    for rname in FOCUS:
        for d,col,values,qcol in [(stages,'stage',['initial'],'q'),
            (within,'contrast',['pooled_r36'],'q'),(stages,'stage',['early','middle','late'],'q'),
            (direct,'pair',['early_minus_middle','early_minus_late','middle_minus_late'],'q_three_pairs_144')]:
            for v in values:
                r=one(primary(d),roi=rname,**{col:v})
                rows.append([roi(rname),LABELS[v],ci(r),count(r['n']),qcell(r,qcol)])
    table('n22_primary_roi_estimates', 'Primary and right-accumbens ROI contrasts. '+ROI_CAP+
          r'BH correction: 48 ROIs separately per stage or within-run contrast; '
          r'144 ROI $\times$ phase-pair tests jointly for direct comparisons.',
          ['ROI','Contrast','Estimate [95\\% CI]','$n$','$q$'],rows,notes=ROI_NOTE)
    for name,d,col,qcol,title in [
        ('stages',stages,'stage','q','All planned stage contrasts'),
        ('within_run',within,'contrast','q','All planned within-run Str2--Str1 contrasts'),
        ('trends',trends,'contrast','q','Exploratory linear trends across runs 3--6'),
        ('direct',direct,'pair','q_three_pairs_144','All direct phase comparisons')]:
        d=primary(d); rows=[]
        assert d.roi.nunique()==48
        for v in d[col].unique():
            block=sub(d,**{col:v}).sort_values('roi'); assert len(block)==48
            rows += [[roi(r.roi),LABELS.get(v,contrast(v)),ci(r),count(r.n),qcell(r,qcol)]
                     for _,r in block.iterrows()]
        family = (r'BH correction jointly over all 144 ROI $\times$ phase-pair tests.' if col=='pair'
                  else r'BH correction over 48 planned ROI slots separately for each contrast.')
        note=ROI_NOTE
        if name=='trends':note+=r' Trend slopes combine prevalence, practice, and familiarity; they are not isolated prevalence effects.'
        table('n22_all_roi_'+name,title+'. '+ROI_CAP+family,
              ['ROI','Contrast','Estimate [95\\% CI]','$n$','$q$'],rows,notes=note)
    specs=read(GLM/'targeted_sensitivity/all_stage_results.csv')
    rows=[]
    for m in MODELS:
        for rn in ROIS:
            r=one(specs,cohort='N22',model=m,policy='observed80',stage='early',roi=rn)
            rows.append([MODEL_NAMES[m],roi(rn),ci(r),count(r.n),qcell(r)])
    table('n22_specifications',r'Early DiD across nine specifications: $N=22$, at least 80\% observed support. '
        r'Pointwise 95\% Student-$t$ CIs; BH over 48 ROIs separately within specification and stage.',
        ['Specification','ROI','Estimate [95\\% CI]','$n$','$q$'],rows,
        notes=ROI_NOTE+r' Precision weighting changes the run combination; RT adjustment changes the estimand. No correction across specifications is claimed.')
    rows=[]
    for cohort,m in [('N22','reference'),('N20','reference'),('N22','censored')]:
        for rn in ROIS:
            counts=[]
            for policy in ['observed80','observed95','complete100']:
                if cohort=='N20' and policy!='observed80':
                    counts.append(r'\textemdash');continue
                r=one(stages,cohort=cohort,model=m,policy=policy,stage='early',roi=rn)
                trace(r,['n']);counts.append(count(r.n))
            rows.append([cohort,MODEL_NAMES[m],roi(rn),*counts])
    table('n22_sensitivity_counts',r'Eligible participant counts for the primary early ROI set at 80\%, 95\%, and 100\% observed support. '
        r'$N=22$ uncensored reference is primary; $N=20$ reference and $N=22$ censoring are sensitivities. '
        r'These are the saved early-stage $n$ values. Stricter coverage is tabulated for $N=22$ only; '
        r'CIs and correction are not applicable to eligibility counts; corresponding estimates have pointwise '
        r'95\% Student-$t$ CIs and 48-slot BH correction per cohort/model/policy/stage.',
        ['Cohort','Model','ROI',r'$n$ (80\%)',r'$n$ (95\%)',r'$n$ (100\%)'],rows)
    rows=[]
    for cohort,m in [('N22','reference'),('N20','reference'),('N22','censored')]:
        for policy,plabel in [('observed80','80\\%'),('observed95','95\\%'),('complete100','100\\%')]:
            if cohort=='N20' and policy!='observed80':continue
            for stage in ['initial','early','middle','late']:
                for rn in ROIS:
                    r=one(stages,cohort=cohort,model=m,policy=policy,stage=stage,roi=rn)
                    rows.append([cohort, 'Reference' if m=='reference' else 'Censored',plabel,
                        roi(rn),LABELS[stage],ci(r),count(r.n),qcell(r)])
    table('n22_sensitivity_matrix',r'Signed estimates for the primary early ROI set under cohort, censoring, and coverage sensitivities. '
        r'$N=22$ uncensored reference with 80\% support is primary; $N=20$ and censoring are sensitivities. '
        r'Pointwise 95\% Student-$t$ CIs; BH over 48 ROIs separately per cohort/model/policy/stage.',
        ['Cohort','Model','Support','ROI','Contrast','Estimate [95\\% CI]','$n$','$q$'],rows,
        notes=ROI_NOTE+r' N=20 and N=22 overlap. See Table~\ref{tab:n22-sensitivity-counts} for coverage-specific counts.')
    influence=read(GLM/'targeted_sensitivity/all_right_accumbens_influence.csv')
    rows=[]
    for cohort in ['N22','N20']:
        for m in MODELS:
            block=sub(influence,cohort=cohort,model=m)
            full=one(block,omitted_subject='none');loo=block[block.omitted_subject!='none']
            assert len(loo)==int(cohort[1:])
            for column in ['mean','q_three_pairs_144']:
                for ix in [loo[column].idxmin(),loo[column].idxmax()]:trace(loo.loc[ix],[column,'n'])
            rows.append([cohort,MODEL_NAMES[m],ci(full),count(full.n),qcell(full,'q_three_pairs_144'),
                f(loo['mean'].min(),signed=True)+' to '+f(loo['mean'].max(),signed=True),
                q(loo.q_three_pairs_144.min())+' to '+q(loo.q_three_pairs_144.max()),count(loo.n.min())])
    table('n22_accumbens_influence',r'Right accumbens early--late full estimates and ranges of saved leave-one-participant-out (LOO) fits. '
        r'$N=22$ primary and overlapping $N=20$ sensitivity; nine specifications; 80\% observed support. '
        r'Full-fit CIs are pointwise 95\% Student-$t$ intervals. Every full/LOO $q$ uses its saved 144-test joint direct-phase BH family.',
        ['Cohort','Specification','Full estimate [95\\% CI]','$n$','Full $q$','LOO estimate range','LOO $q$ range','LOO $n$'],rows,
        notes=r'Ranges report extrema of existing deletion fits, not confidence intervals. RT adjustment and precision weighting change the estimand.')
    return stages


def behavior_tables():
    rt=read(BEH/'rt_comparison/matched_contrasts.csv')
    error=read(BEH/'error_models/matched_contrasts.csv')
    keys=['run2_Str1_minus_Ran','Str1_run2_minus_run1','Str2_run8_minus_run3',
        'run3_Str2_minus_Str1','run4_Str2_minus_Str1','run5_Str2_minus_Str1','run6_Str2_minus_Str1',
        'early_adjusted_change','middle_adjusted_change','late_adjusted_change',
        'early_minus_middle','early_minus_late','middle_minus_late']
    rows=[]
    for outcome,d,where,cols in [('RT',rt,{'policy':'primary'},['geometric_rt_ratio','ratio_ci_low','ratio_ci_high']),
        ('Error',error,{'model':'binomial'},['odds_ratio','odds_ratio_ci_low','odds_ratio_ci_high'])]:
        for c in keys:
            a=one(d,cohort='N22',contrast=c,**where)
            rows.append([outcome,contrast(c),ci(a,*cols,signed=False),qcell(a,'q_bh_24')])
    main_note = BEH_NOTE.replace(
        r'Approximate tests using participant df = 21 ($N=22$) and 19 ($N=20$). '
        r'The samples overlap and comparisons are descriptive. ',
        r'Approximate tests using participant df = 21 ($N=22$). ')
    main_note = main_note.replace('separately by cohort, response policy, and outcome/model',
                                  'separately by response policy and outcome/model')
    table('n22_behavior_contrasts',r'Key behavioral contrasts in the $N=22$ sample. '
        r'RT: geometric ratio from the log-RT mixed model; errors: conditional odds ratio from the binomial mixed model. '
        r'Pointwise 95\% CIs; separate 24-contrast BH families by outcome; imaging coverage not applicable.',
        ['Outcome','Contrast','$N=22$ ratio [95\\% CI]','$q$'],rows,notes=main_note)
    for name,d,where,cols,title in [
        ('rt_lme_primary',rt,{'policy':'primary'},['geometric_rt_ratio','ratio_ci_low','ratio_ci_high'],'Primary log-RT mixed model, all cleaned correct responses'),
        ('rt_lme_min100',rt,{'policy':'min100'},['geometric_rt_ratio','ratio_ci_low','ratio_ci_high'],'Log-RT sensitivity, cleaned correct RT at least 100 ms'),
        ('error_binomial',error,{'model':'binomial'},['odds_ratio','odds_ratio_ci_low','odds_ratio_ci_high'],'Binomial error mixed model, conditional odds ratios'),
        ('error_gaussian',error,{'model':'gaussian'},['difference_pp','ci_low_pp','ci_high_pp'],'Gaussian error sensitivity, percentage-point differences')]:
        rows=[]
        for c in sub(d,cohort='N22',**where).contrast:
            a=one(d,cohort='N22',contrast=c,**where);b=one(d,cohort='N20',contrast=c,**where)
            rows.append([contrast(c),ci(a,*cols,signed=name=='error_gaussian'),qcell(a,'q_bh_24'),
                         ci(b,*cols,signed=name=='error_gaussian'),qcell(b,'q_bh_24')])
        assert len(rows)==24
        table('n22_'+name,title+r'. $N=22$ primary and $N=20$ sensitivity. Pointwise 95\% CIs; '
            r'BH over all 24 contrasts separately by cohort and model/response policy; imaging coverage not applicable.',
            ['Contrast','$N=22$ estimate [95\\% CI]','$q$','$N=20$ estimate [95\\% CI]','$q$'],rows,notes=BEH_NOTE)


def diagnostic_tables():
    wb=read(GLM/'voxelwise/permutation_summary.csv');rows=[]
    for _,r in wb.iterrows():
        trace(r,list(r.index))
        pooled=r.contrast=='pooled_r36_Str2_minus_Str1'
        rows.append([r.cohort,MODEL_NAMES[r.model], 'Pooled Str2--Str1' if pooled else r.contrast.title(),
            count(r.n_subjects),f(r.alpha,4),count(r.n_perm),count(r.positive_voxels),count(r.negative_voxels),
            'Single contrast' if pooled else 'Four stages'])
    table('n22_wholebrain_counts',r'All saved whole-brain permutation summaries: $N=22$ primary uncensored reference and matched sensitivities. '
        r'Two-sided TFCE, voxelwise FWE within contrast; stage family uses Bonferroni across four stages ($\alpha=.0125$), '
        r'pooled within-run contrast has a separate $\alpha=.05$ family per cohort/model. '
        r'Common whole-brain support: 58,550 voxels; ROI coverage and estimate CIs are not applicable to threshold counts.',
        ['Cohort','Model','Contrast','$n$',r'$\alpha$','Permutations','Positive','Negative','Family'],rows,
        notes=r'Counts describe corrected voxels, not clusterwise inference or tests of model differences.')
    motion=read(GLM/'motion_quality/cohort_motion_summary.csv')
    metrics={'n_runs':'Runs','median_run_mean_fd':'Median run mean FD (mm)',
        'max_run_mean_fd':'Maximum run mean FD (mm)', 'n_run_mean_fd_gt02':'Runs with mean FD > 0.2 mm',
        'n_run_mean_fd_gt05':'Runs with mean FD > 0.5 mm','n_runs_gt20pct_fd_gt05':'Runs with >20\\% frames at FD >0.5 mm',
        'n_censored':'Frames meeting censoring rule','n_total_frames':'Total frames',
        'max_run_censored_fraction':'Maximum fraction censored per run'}
    rows=[]
    for col,label in metrics.items():
        vals=[]
        for c in ['N22','N20']:
            r=one(motion,cohort=c);trace(r,[col]);v=r[col];vals.append(count(v) if col.startswith('n_') else f(v))
        rows.append([label,*vals])
    table('n22_motion',r'Descriptive motion quality in the $N=22$ primary and overlapping $N=20$ sensitivity cohorts. '
        r'The reference model retains all frames; censoring is a sensitivity. ROI coverage, CIs, and multiplicity correction are not applicable to these descriptive counts and saved summaries.',
        ['Measure','$N=22$','$N=20$'],rows)
    diag=read(GLM/'design_diagnostics/median_diagnostics_by_run.csv');rows=[]
    for cohort in ['N22','N20']:
        for model in ['reference','censored']:
            for _,r in sub(diag,cohort=cohort,model=model).sort_values('run').iterrows():
                trace(r,list(r.index))
                rows.append([cohort,'Reference' if model=='reference' else 'Censored',count(r.run),
                    count(r['rank']), f(r.Ran_nuisance_adjusted_vif), f(r.Str1_nuisance_adjusted_vif),
                    f(r.Str2_nuisance_adjusted_vif),f(r.adjusted_Str1_Str2_correlation),
                    f(r.Str2_minus_Str1_iid_unit_variance),f(r.Str2_minus_Str1_fullmask_AR_unit_variance)])
    table('n22_design_diagnostics',r'Saved median design diagnostics by run in $N=22$ primary and $N=20$ sensitivity cohorts, reference and censored models. '
        r'VIFs are nuisance-adjusted. The variance columns give Structure~2 $-$ Structure~1 unit-noise design variance '
        r'under iid noise and full-mask fitted AR covariance, respectively; these are not ROI sampling variances or SEs. '
        r'Full-mask AR diagnostics are available only for reference fits. '
        r'ROI coverage thresholds, CIs, and multiplicity correction are not applicable to these descriptive design summaries.',
        ['Cohort','Model','Run','Rank','Random VIF','Str1 VIF','Str2 VIF','Str1/Str2 $r$','iid variance','AR variance'],rows,
        notes=r'Absent conditions or unavailable saved diagnostics are shown as \textemdash. Medians were supplied by the validated task, not recomputed here.')


def save_figure(fig,name):
    for ext in ['pdf','png']:
        p=FIGURES/(name+'.'+ext)
        fig.savefig(p,bbox_inches='tight',metadata=PDF_META if ext=='pdf' else {'Software':'T9 validated manuscript asset formatter'})
        register(p)
        FIGURE_IDS[str(p.relative_to(ROOT))] = sorted(USED_IDS)
    USED_IDS.clear()
    plt.close(fig)


def specifications_display():
    """Plot only saved N=22 estimates/CIs for the fixed six-ROI display set."""
    specs = read(GLM/'targeted_sensitivity/all_stage_results.csv')
    display_rois = ROIS + ['LeftAccumbensArea', 'RightAccumbensArea']
    stages = [('initial', 'Initial Structure 1 − Random'), ('early', 'Early DiD'),
              ('middle', 'Middle DiD'), ('late', 'Late DiD')]
    out = FIGURES/'n22_specifications_display.pdf'
    all_ids = set()
    with PdfPages(out, metadata=PDF_META) as pdf:
        for page, rn in enumerate(display_rois, 1):
            assert not USED_IDS
            saved_rows = []
            fig, axes = plt.subplots(1, 4, figsize=(12.8, 3.8), sharey=True)
            fig.subplots_adjust(left=.23, right=.985, bottom=.23, top=.76, wspace=.23)
            for ax, (stage, title) in zip(axes, stages):
                ax.axvline(0, color='#888888', linewidth=.8, linestyle='--', zorder=0)
                for position, model in enumerate(MODELS):
                    r = one(specs, cohort='N22', model=model, policy='observed80',
                            stage=stage, roi=rn)
                    trace(r, ['mean', 'ci_low', 'ci_high', 'n'])
                    saved_rows.append({
                        'row_key':{k:str(r[k]) for k in ['cohort','model','policy','stage','roi']},
                        'values':{k:float(r[k]) for k in ['mean','ci_low','ci_high','n']}})
                    assert r.n == 22
                    assert all(math.isfinite(r[c]) for c in ['mean', 'ci_low', 'ci_high'])
                    ax.errorbar(r['mean'], position,
                                xerr=[[r['mean'] - r.ci_low], [r.ci_high - r['mean']]],
                                fmt='o', color='#2878A0', markersize=3.5, capsize=2, linewidth=1)
                ax.set_title(title, fontsize=10, pad=8)
                ax.set_xlabel('Signed estimate', fontsize=9)
                ax.set_yticks(range(len(MODELS)))
                ax.set_ylim(len(MODELS)-.5, -.5)
                ax.tick_params(axis='both', labelsize=9)
                ax.grid(axis='x', color='#dddddd', linewidth=.5)
            axes[0].set_yticklabels([MODEL_NAMES[m] for m in MODELS], fontsize=9.5)
            fig.suptitle(roi(rn) + ' — nine specifications', fontsize=13, fontweight='bold', y=.97)
            fig.text(.5, .865, 'N=22; at least 80% observed parcel support; eligible n=22 for every displayed fit',
                     ha='center', fontsize=10)
            fig.text(.5, .09, 'Points: saved estimates; bars: pointwise 95% CIs. Initial and DiD estimates use two-run sums.',
                     ha='center', fontsize=9)
            fig.text(.5, .035, 'BA6 denotes the complete Brodmann parcel. RT adjustment and precision weighting change the estimand.',
                     ha='center', fontsize=9)
            pdf.savefig(fig)
            preview = FIGURES/f'n22_specifications_display_page{page}.png'
            fig.savefig(preview, metadata={'Software':'T9 validated manuscript asset formatter'})
            register(preview)
            FIGURE_IDS[str(preview.relative_to(ROOT))] = sorted(USED_IDS)
            FIGURE_SOURCE_ROWS[str(preview.relative_to(ROOT))] = saved_rows
            all_ids.update(USED_IDS)
            USED_IDS.clear()
            plt.close(fig)
    register(out)
    FIGURE_IDS[str(out.relative_to(ROOT))] = sorted(all_ids)


def components_display():
    """Retain the validated figure's six ROIs and N=22 source rows only."""
    components = read(GLM/'roi_reporting/phase_component_results.csv')
    contrasts = read(GLM/'roi_reporting/stage_results.csv')
    original_inputs = json.loads((GLM/'figures/input_manifest.json').read_text())['inputs']
    for data in [components, contrasts]:
        source = ROOT/data.attrs['source']
        recorded = [v['sha256'] for k, v in original_inputs.items()
                    if k.endswith('/roi_reporting/'+source.name)]
        if recorded != [sha(source)]:
            raise ValueError(f'Original figure input mismatch: {source}')
    focus = ['LeftSMA', 'RightAccumbensArea', 'LeftPutamen', 'LeftCaudate',
             'LeftHippocampus', 'LeftCerebellum']
    stages = ['initial', 'early', 'middle', 'late']
    saved_rows = []
    fig, axes = plt.subplots(3, 4, figsize=(14, 10))
    for i, rn in enumerate(focus):
        left, right = axes[i//2, (i%2)*2:(i%2)*2+2]
        for ax, data, component, color, label in [
            (left, components, 'structured', '#326b9d', 'Structured'),
            (left, components, 'random', '#858585', 'Random'),
            (right, contrasts, None, '#268378', 'N=22')]:
            selected = []
            for stage in stages:
                key = dict(cohort='N22', model='reference', policy='observed80',
                           stage=stage, roi=rn)
                if component is not None:
                    key['component'] = component
                r = one(data, **key)
                trace(r, ['mean', 'ci_low', 'ci_high', 'n'])
                assert r.n >= 2
                selected.append(r)
                saved_rows.append({'source':data.attrs['source'], 'row_key':key,
                    'values':{k:float(r[k]) for k in ['mean','ci_low','ci_high','n']}})
            means = [r['mean'] for r in selected]
            ax.errorbar(range(4), means,
                yerr=[[r['mean']-r.ci_low for r in selected],
                      [r.ci_high-r['mean'] for r in selected]],
                fmt='o-' if component else 'o', ms=3, label=label, color=color)
        right.axvline(.5, color='.7', ls=':')
        left.set_ylabel('Two-run mean coefficient', fontsize=10)
        right.set_ylabel('Signed contrast (two-run sum)', fontsize=10)
        title = 'Left cerebellar cortex' if rn == 'LeftCerebellum' else roi(rn)
        eligible = {int(r['values']['n']) for r in saved_rows if r['row_key']['roi'] == rn}
        assert len(eligible) == 1
        title += f' (n={eligible.pop()})'
        for ax in [left, right]:
            ax.axhline(0, color='.7', lw=.6)
            ax.set_xticks(range(4), [s.title() for s in stages], fontsize=8)
            ax.set_title(title, fontsize=10)
            ax.tick_params(axis='y', labelsize=9)
    axes[0, 0].legend(fontsize=8)
    axes[0, 1].legend(fontsize=8)
    fig.suptitle('N=22 condition components and signed contrasts; pointwise 95% CIs', fontsize=14)
    fig.tight_layout(rect=(0, .035, 1, .98))
    fig.text(.5, .015, 'Uncensored reference; at least 80% observed parcel support. '
             'Contrasts: initial Structure 1 − Random; later difference-in-differences (DiD).',
             ha='center', fontsize=10)
    save_figure(fig, 'n22_fig2_components')
    for ext in ['pdf', 'png']:
        FIGURE_SOURCE_ROWS[f'manuscript/figures/n22_fig2_components.{ext}'] = saved_rows


def wholebrain_display(name, panels):
    """Compose unchanged validated axial PNGs on two matched portrait pages."""
    out=FIGURES/(name+'.pdf')
    with PdfPages(out,metadata=PDF_META) as pdf:
        for page in [1,2]:
            fig,axes=plt.subplots(len(panels),1,figsize=(6.5,9.0))
            fig.subplots_adjust(left=.025,right=.975,bottom=.045,top=.92,hspace=.19)
            for ax,(sample,model,label) in zip(axes,panels):
                path=checked(GLM/f'anatomical_display/{sample}_{model}_axial_page{page}.png')
                ax.imshow(plt.imread(path));ax.axis('off');ax.set_title(label,fontsize=9,fontweight='bold',pad=3)
            fig.suptitle('Pooled runs 3–6 Str2 − Str1: matched axial levels\nSeparate two-sided TFCE/FWE .05 families',fontsize=10,y=.985)
            fig.text(.5,.012,'Neurological orientation; original signed-t scales retained.\nThreshold patterns are descriptive comparisons.',ha='center',fontsize=8)
            pdf.savefig(fig)
            for ext in ['pdf','png']:
                p=FIGURES/f'{name}_page{page}.{ext}'
                fig.savefig(p,metadata=PDF_META if ext=='pdf' else {'Software':'T9 validated manuscript asset formatter'})
                register(p)
            if page==1:
                p=FIGURES/(name+'.png')
                shutil.copyfile(FIGURES/f'{name}_page1.png',p);register(p)
            plt.close(fig)
    register(out)


def figures():
    cells=read(BEH/'rt_lme_N22/primary_cell_estimates.csv')
    for _,r in cells.iterrows():
        trace(r,['geometric_rt_ms','geometric_ci_low_ms','geometric_ci_high_ms','df'])
    fig,(ax,bx)=plt.subplots(2,1,figsize=(9,7),gridspec_kw={'height_ratios':[1,1.1]},layout='constrained')
    colors={'Str1':'#2878A0','Str2':'#D27428','Ran':'#777777'}
    for run in range(1,9):
        p1=1 if run<3 else (0 if run>6 else [.8,.6,.4,.2][run-3])
        for bottom,height,cond in [(0,p1,'Str1'),(p1,1-p1,'Str2')]:
            if height:
                ax.add_patch(Rectangle((run-.37,bottom),.74,height,color=colors[cond]))
                ax.text(run,bottom+height/2,f'{int(round(height*100))}%',color='white',ha='center',va='center',fontsize=9)
        ax.add_patch(Rectangle((run-.37,-.24),.74,.19,color=colors['Ran']))
        ax.text(run,-.145,'Random\n2 × 40 s',color='white',ha='center',va='center',fontsize=8)
    ax.set(xlim=(.4,8.6),ylim=(-.29,1.16),xticks=range(1,9),yticks=[],xlabel='Run')
    ax.set_title('(a) Task schedule: 8 runs × 400 s\nStructured-event prevalence (Str1 / Str2)',loc='left',fontweight='bold',fontsize=11,pad=9)
    ax.spines[['left','bottom']].set_visible(False)
    for c,label in [('Str1','Structure 1'),('Str2','Structure 2'),('Ran','Random')]:
        d=cells[cells.cell.str.endswith('_'+c)].copy()
        runs=d.cell.str.extract(r'r(\d+)_')[0].astype(int).to_numpy()
        y=d.geometric_rt_ms.to_numpy()
        offset={'Str1':-.12,'Str2':0.,'Ran':.12}[c]
        bx.errorbar(runs+offset,y,yerr=[y-d.geometric_ci_low_ms.to_numpy(),d.geometric_ci_high_ms.to_numpy()-y],
            fmt='o-',color=colors[c],label=label,capsize=3,lw=1.6)
    bx.set(xlabel='Run',ylabel='Geometric RT (ms)',xticks=range(1,9))
    bx.set_title('(b) N=22 log-RT mixed-model estimates and pointwise 95% CIs',loc='left',fontweight='bold',fontsize=11)
    bx.legend(ncol=3,frameon=False,loc='upper right')
    save_figure(fig,'n22_fig1_task_behavior')
    for ext in ['pdf','png']:
        src=checked(GLM/f'figures/condition_did_components.{ext}')
        out=FIGURES/f'n22_components_N20_N22.{ext}';shutil.copyfile(src,out);register(out)
    components_display()
    # Retain the complete existing trajectory atlas as a supplementary asset.
    src=checked(GLM/'figures/sequence_trajectories.pdf')
    out=FIGURES/'n22_sequence_trajectories.pdf';shutil.copyfile(src,out);register(out)
    # Preserve all 48 saved ROI pages; the PNG is a first-page preview only.
    src=checked(GLM/'targeted_sensitivity/all_N22_specifications.pdf')
    out=FIGURES/'n22_specifications.pdf';shutil.copyfile(src,out);register(out)
    subprocess.run(['pdftoppm','-f','1','-l','1','-singlefile','-png','-r','120',
                    str(out),str(FIGURES/'n22_specifications')],check=True)
    register(FIGURES/'n22_specifications.png')
    specifications_display()
    # Retain every occupied axial level and the unchanged three-panel supplement.
    wholebrain_display('n22_wholebrain_matched', [
        ('N22','reference','(a) Primary: N=22 reference'),
        ('N20','reference','(b) Sensitivity: N=20 reference'),
        ('N22','censored','(c) Sensitivity: N=22 censored')])
    wholebrain_display('n22_fig3_wholebrain', [
        ('N22','reference','(a) Primary: N=22 reference'),
        ('N22','censored','(b) Sensitivity: N=22 censored')])


def main():
    neural_tables();behavior_tables();diagnostic_tables();figures()
    manifest={'schema_version':1,'generator':{'path':str(Path(__file__).relative_to(ROOT)),
        'sha256':sha(Path(__file__))},'sources':dict(sorted(SOURCES.items())),
        'outputs':dict(sorted(OUTPUTS.items())),
        'fact_ledger':{'path':str(FACT_PATH.relative_to(ROOT)), 'sha256':sha(FACT_PATH) if FACTS else None},
        'table_fact_ids':dict(sorted(TABLE_IDS.items())),
        'figure_fact_ids':dict(sorted(FIGURE_IDS.items())),
        'figure_source_rows':dict(sorted(FIGURE_SOURCE_ROWS.items())),
        'components_display':{
            'path':'manuscript/figures/n22_fig2_components.pdf',
            'cohort':'N22', 'model':'reference', 'policy':'observed80',
            'roi_order':['LeftSMA','RightAccumbensArea','LeftPutamen','LeftCaudate',
                         'LeftHippocampus','LeftCerebellum'],
            'stage_order':['initial','early','middle','late'],
            'component_source':str((GLM/'roi_reporting/phase_component_results.csv').relative_to(ROOT)),
            'contrast_source':str((GLM/'roi_reporting/stage_results.csv').relative_to(ROOT)),
            'original_figure_input_manifest':{
                'path':str((GLM/'figures/input_manifest.json').relative_to(ROOT)),
                'sha256':sha(GLM/'figures/input_manifest.json')},
            'supplement_copy':'manuscript/figures/n22_components_N20_N22.pdf'},
        'specification_display':{
            'path':'manuscript/figures/n22_specifications_display.pdf',
            'source':str((GLM/'targeted_sensitivity/all_stage_results.csv').relative_to(ROOT)),
            'cohort':'N22', 'policy':'observed80',
            'page_rois':ROIS + ['LeftAccumbensArea', 'RightAccumbensArea'],
            'specification_order':MODELS,
            'specification_labels':[MODEL_NAMES[m] for m in MODELS],
            'stage_order':['initial', 'early', 'middle', 'late'],
            'estimate_columns':['mean', 'ci_low', 'ci_high'],
            'preview_pattern':'manuscript/figures/n22_specifications_display_page{page}.png'},
        'editorial_sources':{'docs/REANALYSIS_PLAN_2026-10-18.md':sha(ROOT/'docs/REANALYSIS_PLAN_2026-10-18.md')},
        'notes':['No models, CIs, tests, or corrections recomputed.',
        'Coverage counts are existing eligible n values; LOO ranges select extrema of existing deletion fits.',
        'Figure 1 schematic requires author review of visual representation.',
        'Figure 1 RT conditions have display-only run offsets: Str1 -0.12, Str2 0, Random +0.12.',
        'Main Table 1 and Figures 2/3 report N22 only; complete supplemental behavior tables retain both cohorts.',
        'Figure 2 uses the same six ROIs and N22 source rows as the validated reanalysis figure; the original matched figure is retained byte-identically as n22_components_N20_N22.pdf/png.',
        'Figure 3 has two matched portrait axial pages with vertical N22 reference/censored panels; n22_wholebrain_matched retains the original primary/N20/censored version. PNG aliases are page 1. Original panel color scales retained.',
        'Specification PDF is an unchanged validated 48-page all-ROI plot; its PNG previews page 1 only.',
        'Readable specification display PDF has six pages: left BA6, right BA6, right pallidum, right thalamus, left accumbens, right accumbens. Per-page PNGs preview the same pages.',
        'Specification display uses saved N22 estimates and pointwise 95% CIs at 80% observed support, nine specifications in table order and four stages; no estimates or intervals recomputed.',
        'Specification ledger facts cover early-stage rows only. All four plotted stages have exact CSV row keys and saved values in figure_source_rows; no missing ledger IDs are invented.',
        'Manifest excludes its own hash to avoid self-reference.']}
    p=TABLES/'n22_assets_manifest.json';p.write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
    print(f'Wrote {len(OUTPUTS)} assets; verified {len(SOURCES)} result sources.')


if __name__=='__main__':
    main()
