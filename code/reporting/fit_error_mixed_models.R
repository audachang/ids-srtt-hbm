#!/usr/bin/env Rscript
# Binomial error GLMM plus a denominator-weighted Gaussian LMM sensitivity.
suppressPackageStartupMessages(library(lme4))
suppressPackageStartupMessages(library(jsonlite))
args <- commandArgs(trailingOnly=TRUE)
root <- if(length(args)) normalizePath(args[1],mustWork=TRUE) else normalizePath('.')
B <- if(length(args)>=2) as.integer(args[2]) else 999L
cores <- if(length(args)>=3) as.integer(args[3]) else 8L
input <- file.path(root,'results/behavioral/block_errors_20261013')
out <- file.path(input,'mixed_model');dir.create(out,recursive=TRUE,showWarnings=FALSE)
spec <- list(role='Exploratory error mixed-effects analysis',
 outcome='Incorrect button presses or omissions / all logged trials; no RT exclusion',
 primary='Binomial logit GLMM on participant/block/condition counts, Laplace maximum likelihood (nAGQ=1)',
 fixed='Twenty observed run-condition cells, identical to the RT contrast design',
 random='Independent participant intercept and Str1/Str2 slopes; nested participant-run and participant-run-block intercepts',
 dependence='Random block/run effects model clustering; no separate within-block AR(1) process. Whole-participant bootstrap preserves the observed within-person dependence.',
 gaussian_sensitivity='Gaussian LMM of block error proportions, weighted by trial denominator, fitted by REML. The block variance is zero in the full hierarchy; the equivalent reduced model retains participant intercept/slopes and participant-run intercepts.',
 contrasts='The same 24 contrast weights used for RT, imported unchanged',
 bootstrap=list(unit='Participant, with complete blocks/runs retained and duplicated participants relabeled independently',replicates=B,seed=20261013,cores=cores),
 inference='Covariance from participant-bootstrap fixed coefficients; approximate t inference with df=19, BH within the 24 error contrasts separately for GLMM and Gaussian LMM. Native model-based GLMM z and Gaussian t results are also exported.',
 intervals='Pointwise percentile participant-bootstrap intervals, unadjusted for multiplicity; conditional odds ratios for GLMM contrasts and percentage-point differences for Gaussian contrasts',
 plot='GLMM population-average error probabilities integrate the independent normal random-effect distributions; intervals incorporate fixed effects and variance components from participant-bootstrap refits',
 calibration='Participant-based t tests are an exploratory approximation, not exact finite-sample calibration. Intermixing, sequence exposure and practice covary.',
 optimizers='bobyqa, theta-only initialization preserving the initial PIRLS stage; nloptwrap retry on numerical nonconvergence. Singular bootstrap variance estimates are permitted and recorded.')
write_json(spec,file.path(out,'analysis_spec.json'),pretty=TRUE,auto_unbox=TRUE)
d<-read.csv(file.path(input,'participant_block_error_rates.csv'),stringsAsFactors=FALSE)
d$subject<-factor(d$subject);d$cell<-factor(sprintf('r%02d_%s',d$session,d$rule_group))
d$subject_run<-factor(paste(d$subject,d$session,sep='_'));d$block_id<-factor(paste(d$subject,d$block,sep='_'))
d$is_str1<-as.integer(d$rule_group=='Str1');d$is_str2<-as.integer(d$rule_group=='Str2')
stopifnot(nrow(d)==1760,nlevels(d$subject)==20,nlevels(d$cell)==20,
 sum(d$n_trials)==51200,sum(d$n_errors)==1425,all(d$n_errors+d$n_correct==d$n_trials),
 qr(model.matrix(~0+cell,d))$rank==20)
primary_formula<-cbind(n_errors,n_trials-n_errors)~0+cell+(1+is_str1+is_str2||subject)+(1|subject_run)+(1|block_id)
gaussian_formula<-error_rate~0+cell+(1+is_str1+is_str2||subject)+(1|subject_run)
control_glmm<-glmerControl(optimizer='bobyqa',optCtrl=list(maxfun=200000,rhoend=1e-8))
control_lmm<-lmerControl(optimizer='bobyqa',optCtrl=list(maxfun=200000))
primary_cache<-file.path(out,'initial_binomial_fit.rds')
if(file.exists(primary_cache)){
 primary<-readRDS(primary_cache)
 stopifnot(inherits(primary,'glmerMod'),identical(family(primary)$family,'binomial'),
  identical(family(primary)$link,'logit'),
  all(model.response(model.frame(primary))==cbind(d$n_errors,d$n_trials-d$n_errors)))
 for(col in c('subject','cell','subject_run','block_id','is_str1','is_str2'))
  stopifnot(identical(as.character(model.frame(primary)[[col]]),as.character(d[[col]])))
}else primary<-glmer(primary_formula,data=d,family=binomial,control=control_glmm)
gaussian<-lmer(gaussian_formula,data=d,weights=n_trials,REML=TRUE,control=control_lmm)
gaussian_full<-lmer(error_rate~0+cell+(1+is_str1+is_str2||subject)+(1|subject_run)+(1|block_id),
 data=d,weights=n_trials,REML=TRUE,control=control_lmm)
full_variances<-as.data.frame(VarCorr(gaussian_full))
gaussian_block_variance<-full_variances$vcov[full_variances$grp=='block_id']
stopifnot(length(gaussian_block_variance)==1,gaussian_block_variance<1e-10,
 max(abs(fixef(gaussian_full)-fixef(gaussian)))<1e-6)
saveRDS(gaussian_full,file.path(out,'gaussian_full_hierarchy_model.rds'))
write.csv(full_variances,file.path(out,'gaussian_full_variance_components.csv'),row.names=FALSE)
stopifnot(primary@optinfo$conv$opt==0,!isSingular(primary),
 is.null(primary@optinfo$conv$lme4$messages),gaussian@optinfo$conv$opt==0)
for(id in c('binomial','gaussian')){
 fit<-if(id=='binomial')primary else gaussian
 saveRDS(fit,file.path(out,paste0(id,'_model.rds')))
 writeLines(capture.output(summary(fit)),file.path(out,paste0(id,'_summary.txt')))
 write.csv(data.frame(term=names(fixef(fit)),estimate=unname(fixef(fit)),native_se=sqrt(diag(vcov(fit)))),
           file.path(out,paste0(id,'_fixed_coefficients.csv')),row.names=FALSE)
 write.csv(as.matrix(vcov(fit)),file.path(out,paste0(id,'_fixed_covariance_native.csv')))
 write.csv(as.data.frame(VarCorr(fit)),file.path(out,paste0(id,'_variance_components.csv')),row.names=FALSE)
 diagnostics<-d[c('subject','session','block','rule_group','n_trials','n_errors','n_correct')]
 diagnostics$conditional_prediction<-fitted(fit)
 diagnostics$pearson_residual<-residuals(fit,type='pearson')
 write.csv(diagnostics,file.path(out,paste0(id,'_block_diagnostics.csv')),row.names=FALSE)
}
weights_path<-file.path(root,'results/behavioral/fmri_rt_20261012/lme/min100_contrast_weights.csv')
w<-read.csv(weights_path,check.names=FALSE);terms<-names(fixef(primary))
stopifnot(nrow(w)==24,setequal(names(w)[-1],terms),identical(names(fixef(gaussian)),terms))
write.csv(w,file.path(out,'contrast_weights.csv'),row.names=FALSE)
cellmeta<-unique(d[c('cell','session','rule_group')]);cellmeta<-cellmeta[order(cellmeta$cell),]
write.csv(cellmeta,file.path(out,'cell_metadata.csv'),row.names=FALSE)
# A separately optimized full-data fit verifies the fixed effects and likelihood.
check<-glmer(primary_formula,data=d,family=binomial,start=list(theta=getME(primary,'theta')),
 control=glmerControl(optimizer='nloptwrap',optCtrl=list(maxeval=200000,xtol_abs=1e-10,ftol_abs=1e-10)))
optimizer_check<-list(max_beta_difference=max(abs(fixef(primary)-fixef(check))),
 log_likelihood_difference=abs(as.numeric(logLik(primary))-as.numeric(logLik(check))),
 messages=check@optinfo$conv$lme4$messages)
write_json(optimizer_check,file.path(out,'optimizer_agreement.json'),pretty=TRUE,auto_unbox=TRUE)
stopifnot(optimizer_check$max_beta_difference<.003,optimizer_check$log_likelihood_difference<.001)
subjects<-levels(d$subject);set.seed(20261013)
draws<-matrix(sample.int(length(subjects),B*length(subjects),replace=TRUE),nrow=B,byrow=TRUE)
write.csv(data.frame(replicate=seq_len(B),draws),file.path(out,'bootstrap_subject_draws.csv'),row.names=FALSE)
variance_summary<-function(fit){
 v<-as.data.frame(VarCorr(fit));v<-v[is.na(v$var2),]
 c(intercept_variance=sum(v$vcov[v$var1=='(Intercept)' & v$grp!='Residual']),
   str1_variance=sum(v$vcov[v$var1=='is_str1']),str2_variance=sum(v$vcov[v$var1=='is_str2']))
}
write.csv(data.frame(component=names(variance_summary(primary)),variance=variance_summary(primary)),
 file.path(out,'binomial_marginal_variances.csv'),row.names=FALSE)
fit_checked<-function(expr){
 warnings<-character();messages<-character()
 fit<-withCallingHandlers(tryCatch(force(expr),error=function(e)e),
  warning=function(w){warnings<<-c(warnings,conditionMessage(w));invokeRestart('muffleWarning')},
  message=function(m){messages<<-c(messages,conditionMessage(m));invokeRestart('muffleMessage')})
 if(inherits(fit,'error'))return(list(ok=FALSE,error=conditionMessage(fit),warnings=warnings,messages=messages))
 conv<-fit@optinfo$conv$lme4$messages
 bad<-conv[!grepl('^boundary \\(singular\\)',conv)]
 list(ok=fit@optinfo$conv$opt==0 && !length(bad) && all(is.finite(fixef(fit))),
      fit=fit,error=paste(bad,collapse='; '),warnings=warnings,messages=messages)
}
fit_bootstrap<-function(i){
 pieces<-lapply(seq_along(subjects),function(j){
  p<-d[d$subject==subjects[draws[i,j]],]
  p$subject<-sprintf('boot%02d',j);p$subject_run<-paste(p$subject,p$session,sep='_')
  p$block_id<-paste(p$subject,p$block,sep='_');p
 })
 b<-do.call(rbind,pieces)
 for(col in c('subject','subject_run','block_id'))b[[col]]<-factor(b[[col]])
 a<-fit_checked(glmer(primary_formula,data=b,family=binomial,
      start=list(theta=getME(primary,'theta')),control=control_glmm))
 retried<-FALSE
 if(!a$ok){
  retried<-TRUE
  a<-fit_checked(glmer(primary_formula,data=b,family=binomial,start=list(theta=getME(primary,'theta')),
       control=glmerControl(optimizer='nloptwrap',optCtrl=list(maxeval=200000,xtol_abs=1e-8,ftol_abs=1e-8))))
 }
 g<-fit_checked(lmer(gaussian_formula,data=b,weights=n_trials,REML=TRUE,control=control_lmm))
 result<-list(replicate=i,binomial_ok=a$ok,gaussian_ok=g$ok,binomial_retry=retried,
     binomial_error=a$error,gaussian_error=g$error,
     binomial_warnings=a$warnings,gaussian_warnings=g$warnings)
 if(a$ok){result$binomial_beta<-unname(fixef(a$fit)[terms]);result$binomial_variances<-unname(variance_summary(a$fit));result$binomial_singular<-isSingular(a$fit)}
 if(g$ok){result$gaussian_beta<-unname(fixef(g$fit)[terms]);result$gaussian_singular<-isSingular(g$fit)}
 result
}
checkpoint<-file.path(out,'bootstrap_checkpoint.rds')
results<-if(file.exists(checkpoint))readRDS(checkpoint)else vector('list',B)
stopifnot(length(results)==B)
for(start in seq.int(1L,B,by=64L)){
 ids<-seq.int(start,min(B,start+63L));todo<-ids[vapply(results[ids],is.null,logical(1))]
 if(length(todo)){
  got<-parallel::mclapply(todo,fit_bootstrap,mc.cores=cores,mc.preschedule=FALSE,mc.set.seed=FALSE)
  for(j in seq_along(todo))results[[todo[j]]]<-got[[j]]
  saveRDS(results,checkpoint)
 }
 done<-results[!vapply(results,is.null,logical(1))]
 cat(sprintf('Completed %d/%d participant refits; binomial valid %d; Gaussian valid %d\n',
 length(done),B,sum(vapply(done,function(x)isTRUE(x$binomial_ok),logical(1))),
 sum(vapply(done,function(x)isTRUE(x$gaussian_ok),logical(1)))))
 flush.console()
}
status<-data.frame(replicate=seq_len(B),
 binomial_ok=vapply(results,function(x)isTRUE(x$binomial_ok),logical(1)),
 gaussian_ok=vapply(results,function(x)isTRUE(x$gaussian_ok),logical(1)),
 binomial_retry=vapply(results,function(x)isTRUE(x$binomial_retry),logical(1)),
 binomial_singular=vapply(results,function(x)isTRUE(x$binomial_singular),logical(1)),
 gaussian_singular=vapply(results,function(x)isTRUE(x$gaussian_singular),logical(1)))
write.csv(status,file.path(out,'bootstrap_status.csv'),row.names=FALSE)
write_json(lapply(results,function(x)x[c('replicate','binomial_ok','gaussian_ok','binomial_retry','binomial_error','gaussian_error','binomial_warnings','gaussian_warnings')]),
 file.path(out,'bootstrap_messages.json'),pretty=TRUE,auto_unbox=TRUE)
for(id in c('binomial','gaussian')){
 ok<-status[[paste0(id,'_ok')]]
 beta<-do.call(rbind,lapply(results[ok],function(x)x[[paste0(id,'_beta')]]));colnames(beta)<-terms
 write.csv(data.frame(replicate=status$replicate[ok],beta,check.names=FALSE),
  file.path(out,paste0(id,'_bootstrap_fixed_coefficients.csv')),row.names=FALSE)
 if(id=='binomial'){
  v<-do.call(rbind,lapply(results[ok],function(x)x$binomial_variances));colnames(v)<-names(variance_summary(primary))
  write.csv(data.frame(replicate=status$replicate[ok],v),file.path(out,'binomial_bootstrap_marginal_variances.csv'),row.names=FALSE)
 }
}
fit_status<-list(status='completed',n_subjects=20,n_rows=1760,n_trials=51200,n_errors=1425,
 n_observed_cells=20,n_contrasts=24,bootstrap_requested=B,bootstrap_binomial_valid=sum(status$binomial_ok),
 bootstrap_gaussian_valid=sum(status$gaussian_ok),bootstrap_binomial_retries=sum(status$binomial_retry),
 bootstrap_binomial_singular=sum(status$binomial_singular & status$binomial_ok),
 bootstrap_gaussian_singular=sum(status$gaussian_singular & status$gaussian_ok),
 primary_singular=isSingular(primary),gaussian_singular=isSingular(gaussian),
 primary_log_likelihood=as.numeric(logLik(primary)),gaussian_log_likelihood=as.numeric(logLik(gaussian)),
 primary_convergence=primary@optinfo$conv,gaussian_convergence=gaussian@optinfo$conv,
 gaussian_block_variance_in_full_model=gaussian_block_variance,optimizer_agreement=optimizer_check)
write_json(fit_status,file.path(out,'fit_status.json'),pretty=TRUE,auto_unbox=TRUE)
writeLines(capture.output(sessionInfo()),file.path(out,'R_session_info.txt'))
print(fit_status)
stopifnot(sum(status$binomial_ok)>=.95*B,sum(status$gaussian_ok)>=.95*B)
