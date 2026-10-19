#!/usr/bin/env Rscript
suppressPackageStartupMessages(library(lme4))
suppressPackageStartupMessages(library(jsonlite))
args<-commandArgs(trailingOnly=TRUE)
input<-normalizePath(args[1],mustWork=TRUE);root<-normalizePath(args[2],mustWork=TRUE)
B<-if(length(args)>2)as.integer(args[3])else 999L
cores<-if(length(args)>3)as.integer(args[4])else 6L
out<-file.path(input,'glmm');dir.create(out,recursive=TRUE,showWarnings=FALSE)
d<-read.csv(file.path(input,'participant_block_anticipation.csv'),stringsAsFactors=FALSE)
d$subject<-factor(d$subject);d$cell<-factor(sprintf('r%02d_%s',d$session,d$rule_group))
d$subject_run<-factor(paste(d$subject,d$session,sep='_'));d$block_id<-factor(paste(d$subject,d$block,sep='_'))
d$is_str1<-as.integer(d$rule_group=='Str1');d$is_str2<-as.integer(d$rule_group=='Str2')
stopifnot(nrow(d)==1760,nlevels(d$subject)==20,nlevels(d$cell)==20,sum(d$n_trials)==51200)
f<-cbind(n_anticipatory,n_trials-n_anticipatory)~0+cell+(1+is_str1+is_str2||subject)+(1|subject_run)+(1|block_id)
control<-glmerControl(optimizer='bobyqa',optCtrl=list(maxfun=200000,rhoend=1e-8))
warnings<-character()
cache<-file.path(out,'initial_bobyqa_fit.rds')
if(file.exists(cache)) fit<-readRDS(cache) else {
 fit<-withCallingHandlers(glmer(f,data=d,family=binomial,control=control),warning=function(w){warnings<<-c(warnings,conditionMessage(w));invokeRestart('muffleWarning')})
 saveRDS(fit,cache)
}
stopifnot(all(model.response(model.frame(fit))==cbind(d$n_anticipatory,d$n_trials-d$n_anticipatory)))
retry_warnings<-character()
retry<-withCallingHandlers(glmer(f,data=d,family=binomial,start=list(theta=getME(fit,'theta')),
 control=glmerControl(optimizer='nloptwrap',optCtrl=list(maxeval=200000,ftol_abs=1e-10,xtol_abs=1e-10))),
 warning=function(w){retry_warnings<<-c(retry_warnings,conditionMessage(w));invokeRestart('muffleWarning')})
saveRDS(retry,file.path(out,'nloptwrap_check.rds'))
agreement<-list(max_abs_fixed_difference=max(abs(fixef(fit)-fixef(retry))),max_abs_fitted_probability_difference=max(abs(fitted(fit)-fitted(retry))),loglik_difference=as.numeric(logLik(fit)-logLik(retry)),bobyqa_messages=fit@optinfo$conv$lme4$messages,nloptwrap_messages=retry@optinfo$conv$lme4$messages,nloptwrap_warnings=retry_warnings)
write_json(agreement,file.path(out,'optimizer_agreement.json'),pretty=TRUE,auto_unbox=TRUE)
if(agreement$max_abs_fixed_difference>=.02 || agreement$max_abs_fitted_probability_difference>=.005 || !is.null(retry@optinfo$conv$lme4$messages)) {
 cells<-aggregate(cbind(n_trials,n_anticipatory)~cell,data=d,sum)
 write.csv(cells,file.path(out,'cell_event_counts.csv'),row.names=FALSE)
 status<-list(status='model_not_identifiable',valid_for_inference=FALSE,
 reason='Zero-event random cells in runs 3 and 6 cause separation; optimizers have nearly identical likelihood/prediction but disagree on finite fixed coefficients and report Hessian/identifiability warnings.',
 agreement=agreement,bootstrap_requested=B,completed_valid_bootstrap_replicates=0,
 action='No odds-ratio inference or bootstrap covariance is reported from this unstable full model. Do not interpret saved attempt coefficients as valid estimates.')
 write_json(status,file.path(out,'status.json'),pretty=TRUE,auto_unbox=TRUE)
 write_json(list(outcome='0<RT<100ms responses / all trials',fixed='20 run-condition cells as requested',random='Independent participant intercept and Str1/Str2 slopes; participant-run and participant-block intercepts',attempted_optimizers=c('bobyqa','nloptwrap'),valid_for_inference=FALSE),file.path(out,'analysis_spec.json'),pretty=TRUE,auto_unbox=TRUE)
 writeLines(capture.output(sessionInfo()),file.path(out,'R_session_info.txt'))
 writeLines(c('# Anticipation GLMM: not identifiable',
 '\nThe specified full model was attempted with the same fixed/random structure as the error model. Two random cells have zero events (runs 3 and 6). BOBYQA reported near-unidentifiability; nloptwrap reported a degenerate Hessian. Fixed coefficients differ by over two log-odds units despite essentially identical fitted probabilities and likelihood. No conditional OR, p value, or bootstrap CI is accepted from this model.',
 '\nThe saved model/coefficient files document failed attempts; they must not be used for manuscript inference. The generated bootstrap draws from the initial attempt produced no accepted final bootstrap result. Raw rates, accuracy, positions, and separately labeled descriptive participant-bootstrap probability differences remain available. A regularized/binomial model with an altered fixed structure would require a new analysis specification.'),file.path(out,'REPORT.md'))
 cat('Full anticipation GLMM is not identifiable; failure documented.\n');quit(status=0)
}

# Prefer the optimizer with no convergence message; preserve both fits.
if(is.null(retry@optinfo$conv$lme4$messages))fit<-retry

saveRDS(fit,file.path(out,'anticipation_model.rds'));writeLines(capture.output(summary(fit)),file.path(out,'summary.txt'))
write.csv(as.data.frame(VarCorr(fit)),file.path(out,'variance_components.csv'),row.names=FALSE)
co<-fixef(fit);write.csv(data.frame(term=names(co),estimate=co,se=sqrt(diag(vcov(fit)))),file.path(out,'fixed_coefficients.csv'),row.names=FALSE)
write.csv(as.matrix(vcov(fit)),file.path(out,'native_covariance.csv'))
weights_path<-file.path(root,'results/behavioral/block_errors_20261013/mixed_model/contrast_weights.csv')
w<-read.csv(weights_path,check.names=FALSE);K<-as.matrix(w[,names(co)]);rownames(K)<-w$contrast
stopifnot(nrow(K)==24,ncol(K)==20)
write.csv(data.frame(contrast=rownames(K),K,check.names=FALSE),file.path(out,'contrast_weights.csv'),row.names=FALSE)
status<-list(status='initial_fit_complete',singular=isSingular(fit),optimizer_code=fit@optinfo$conv$opt,messages=fit@optinfo$conv$lme4$messages,warnings=warnings,B=B,seed=20261018)
write_json(status,file.path(out,'status.json'),pretty=TRUE,auto_unbox=TRUE)
set.seed(20261018);subjects<-levels(d$subject);draws<-replicate(B,sample(subjects,length(subjects),replace=TRUE),simplify=FALSE)
write.csv(do.call(rbind,lapply(seq_len(B),function(i)data.frame(replicate=i,slot=seq_along(subjects),subject=draws[[i]]))),file.path(out,'bootstrap_draws.csv'),row.names=FALSE)
one<-function(i){
 dd<-do.call(rbind,lapply(seq_along(subjects),function(j){
  x<-d[as.character(d$subject)==draws[[i]][j],];x$subject<-sprintf('boot%02d',j)
  x$subject_run<-paste(x$subject,x$session,sep='_');x$block_id<-paste(x$subject,x$block,sep='_');x
 }))
 for(c in c('subject','subject_run','block_id'))dd[[c]]<-factor(dd[[c]])
 ww<-character()
 obj<-withCallingHandlers(tryCatch(glmer(f,data=dd,family=binomial,start=list(theta=getME(fit,'theta')),control=control),error=function(e)e),warning=function(w){ww<<-c(ww,conditionMessage(w));invokeRestart('muffleWarning')})
 if(inherits(obj,'error'))return(list(ok=FALSE,error=conditionMessage(obj),warnings=ww))
 msg<-obj@optinfo$conv$lme4$messages
 singular_only<-is.null(msg)||all(grepl('boundary (singular)',msg,fixed=TRUE))
 list(ok=obj@optinfo$conv$opt==0 && singular_only,coef=fixef(obj),singular=isSingular(obj),messages=msg,warnings=ww)
}
# Deterministic pre-generated participant draws make results independent of fork scheduling.
bootdir<-file.path(out,'bootstrap_runs');dir.create(bootdir,showWarnings=FALSE)
res<-parallel::mclapply(seq_len(B),function(i){
 path<-file.path(bootdir,sprintf('%04d.rds',i))
 if(file.exists(path))return(readRDS(path))
 z<-one(i);saveRDS(z,path);cat(sprintf('Bootstrap %d/%d complete; converged=%s\n',i,B,z$ok));z
},mc.cores=cores,mc.preschedule=FALSE,mc.set.seed=FALSE)
saveRDS(res,file.path(out,'bootstrap_results.rds'))
ok<-vapply(res,function(x)isTRUE(x$ok),logical(1));status$n_bootstrap_converged<-sum(ok);status$n_bootstrap_failed<-sum(!ok)
write.csv(data.frame(replicate=seq_len(B),converged=ok,singular=vapply(res,function(x)isTRUE(x$singular),logical(1))),file.path(out,'bootstrap_status.csv'),row.names=FALSE)
write_json(res[!ok],file.path(out,'bootstrap_failures.json'),pretty=TRUE,auto_unbox=TRUE)
stopifnot(sum(ok)>=.9*B)
boots<-do.call(rbind,lapply(res[ok],function(x)x$coef[names(co)]));write.csv(boots,file.path(out,'bootstrap_coefficients.csv'),row.names=FALSE)
vc<-cov(boots);write.csv(vc,file.path(out,'bootstrap_covariance.csv'))
rows<-lapply(seq_len(nrow(K)),function(i){
 v<-K[i,];est<-sum(v*co);se<-sqrt(drop(v%*%vc%*%v));ci<-quantile(drop(boots%*%v),c(.025,.975));native_se<-sqrt(drop(v%*%as.matrix(vcov(fit))%*%v))
 data.frame(contrast=rownames(K)[i],log_odds_difference=est,se_bootstrap=se,df=19,t=est/se,p=2*pt(-abs(est/se),19),ci_low=ci[1],ci_high=ci[2],odds_ratio=exp(est),or_ci_low=exp(ci[1]),or_ci_high=exp(ci[2]),native_z=est/native_se,native_p=2*pnorm(-abs(est/native_se)))
})
rows<-do.call(rbind,rows);rows$q_bh24<-p.adjust(rows$p,'BH');write.csv(rows,file.path(out,'contrasts.csv'),row.names=FALSE)
status$status<-'complete';status$n_significant_bh24<-sum(rows$q_bh24<=.05)
write_json(status,file.path(out,'status.json'),pretty=TRUE,auto_unbox=TRUE)
write_json(list(outcome='Any logged 0<RT<100ms response / all logged trials, including incorrect responses',fixed='Twenty observed run-condition cells',random='Independent participant intercept and Str1/Str2 slopes, nested participant-run and participant-block intercepts; same structure as error GLMM',inference='Whole-participant bootstrap covariance, approximate t tests using df=19, BH across 24 contrasts; pointwise bootstrap percentile CIs and conditional odds ratios',bootstrap=list(replicates=B,seed=20261018,cores=cores),contrast_weights_source=weights_path),file.path(out,'analysis_spec.json'),pretty=TRUE,auto_unbox=TRUE)
writeLines(capture.output(sessionInfo()),file.path(out,'R_session_info.txt'))
writeLines(c('# T7a anticipation GLMM',sprintf('\nConverged participant bootstraps: %d/%d. Singular full fit: %s. %d/24 BH-significant contrasts.',sum(ok),B,isSingular(fit),sum(rows$q_bh24<=.05)),'\nSame fixed/random structure as the historical error GLMM. Pointwise intervals are participant-bootstrap percentile intervals. Odds ratios are conditional; the t tests using df=19 are approximate. Full variance and convergence diagnostics are retained.'),file.path(out,'REPORT.md'))
print(status)
