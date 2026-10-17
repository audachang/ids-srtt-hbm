#!/usr/bin/env Rscript
# Trial-level log-RT LME with subject, run and block effects and within-block AR(1).
suppressPackageStartupMessages(library(nlme))
suppressPackageStartupMessages(library(jsonlite))
args <- commandArgs(trailingOnly=TRUE)
if (!length(args) %in% c(2,3)) stop('Usage: fit_fmri_rt_lme.R BEHAVIOR_OUTPUT OUTPUT_DIR [--reuse]')
reuse <- length(args)==3 && args[3]=='--reuse'
input <- normalizePath(args[1],mustWork=TRUE);out <- args[2]
dir.create(out,recursive=TRUE,showWarnings=FALSE)
spec <- list(role='Exploratory trial-level LME requested after descriptive inspection',
  outcome='Natural log of correct cleaned RT in milliseconds',
  fixed='Twenty observed run-condition cells; absent Str2 in runs 1-2 and Str1 in runs 7-8 are not fabricated',
  random='Independent subject intercept and Str1/Str2 slopes; nested subject-run and subject-run-block intercepts',
  residual='AR(1) within original block trial indices; condition-specific residual SD',
  estimation='REML; pointwise t intervals and contrast tests use min(nlme denominator DF, number of independent participants minus one)=19; native package DF also exported. This is an exploratory conservative approximation, not an exact finite-sample calibration.',
  inference='BH over all 24 prespecified RT contrasts, not over the descriptive cell means',
  contrasts=c('12 within-run sequence-minus-random','Str1 run2-minus-run1','Str2 run8-minus-run3',
              '4 Str2-minus-Str1 comparisons in runs3-6','3 adjusted change phases','3 direct change-phase pairs'),
  multiplicity='All 24 contrast slots retained; primary 0<RT<1-second cleaning and min100ms sensitivity corrected separately',
  cleaning='Uses previously reconstructed trimr-cleaned trials; additional min100ms sensitivity',
  backtransformation='Fixed-effect geometric RT at random effects zero; not an arithmetic mean or marginal mean integrating random effects',
  fallback='If AR1/heteroscedastic fit fails, report failure and evaluate a separately labeled homoscedastic independent-residual LME; no silent replacement')
write_json(spec,file.path(out,'analysis_spec.json'),pretty=TRUE,auto_unbox=TRUE)
raw <- read.csv(file.path(input,'cleaned_trials.csv'),stringsAsFactors=FALSE)
raw$subject <- factor(raw$subject)
raw$run <- factor(raw$session,levels=1:8)
raw$condition <- factor(raw$rule_group,levels=c('Ran','Str1','Str2'))
raw$cell <- factor(sprintf('r%02d_%s',raw$session,raw$rule_group))
raw$subject_run <- factor(paste(raw$subject,raw$session,sep='_'))
raw$block_id <- factor(paste(raw$subject,raw$session,raw$block,sep='_'))
raw$is_str1 <- as.numeric(raw$condition=='Str1')
raw$is_str2 <- as.numeric(raw$condition=='Str2')
raw$log_rt <- log(raw$p_response_rt*1000)
raw <- raw[order(raw$subject,raw$session,raw$trial_in_run),]
stopifnot(nlevels(raw$subject)==20,nlevels(raw$cell)==20,
          all(is.finite(raw$log_rt)),!anyDuplicated(raw[c('subject','session','trial_in_run')]))
cellmeta <- unique(raw[c('cell','session','condition')]);cellmeta <- cellmeta[order(cellmeta$cell),]
write.csv(cellmeta,file.path(out,'cell_metadata.csv'),row.names=FALSE)
control <- lmeControl(maxIter=100,msMaxIter=300,msMaxEval=600,opt='nlminb',
                       tolerance=1e-6,niterEM=0,returnObject=FALSE,apVar=FALSE)
fit_one <- function(d,id) {
  warnings <- character()
  cached <- file.path(out,paste0(id,'_model.rds'))
  if (reuse && file.exists(cached)) {
    result <- readRDS(cached)
    saved <- getData(result)
    columns <- c('subject','session','trial_in_run','rule_group','p_response_rt','log_rt','cell','subject_run','block_id')
    stopifnot(inherits(result,'lme'),nrow(saved)==nrow(d),
      identical(saved[,columns],d[,columns]),identical(as.character(formula(result)),c('~','log_rt','0 + cell')),
      inherits(result$modelStruct$corStruct,'corARMA'),inherits(result$modelStruct$varStruct,'varIdent'),
      identical(result$method,'REML'))
    cat('Reusing verified completed REML fit:',id,'\n')
  } else result <- withCallingHandlers(tryCatch(lme(log_rt~0+cell,
    random=list(subject=pdDiag(~1+is_str1+is_str2),subject_run=pdDiag(~1),block_id=pdDiag(~1)),
    correlation=corAR1(value=.2,form=~trial_in_run|subject/subject_run/block_id),
    weights=varIdent(form=~1|condition),data=d,method='REML',na.action=na.fail,
    control=control,keep.data=TRUE),error=function(e)e),
    warning=function(w){warnings<<-c(warnings,conditionMessage(w));invokeRestart('muffleWarning')})
  if (inherits(result,'error')) {
    write_json(list(status='failed',message=conditionMessage(result),warnings=warnings),
      file.path(out,paste0(id,'_status.json')),pretty=TRUE,auto_unbox=TRUE)
    stop(paste('LME failed',id,conditionMessage(result)))
  }
  fit <- result;saveRDS(fit,file.path(out,paste0(id,'_model.rds')))
  writeLines(capture.output(summary(fit)),file.path(out,paste0(id,'_summary.txt')))
  coefficients <- as.data.frame(summary(fit)$tTable);coefficients$term <- rownames(coefficients)
  write.csv(coefficients,file.path(out,paste0(id,'_fixed_coefficients.csv')),row.names=FALSE)
  vc <- vcov(fit);write.csv(vc,file.path(out,paste0(id,'_fixed_covariance.csv')))
  tt <- summary(fit)$tTable
  cellrows <- data.frame(policy=id,cell=gsub('^cell','',rownames(tt)),
     log_rt=tt[,'Value'],se=tt[,'Std.Error'],df_nlme=tt[,'DF'],df=pmin(tt[,'DF'],nlevels(d$subject)-1))
  half <- qt(.975,cellrows$df)*cellrows$se
  cellrows$log_ci_low <- cellrows$log_rt-half;cellrows$log_ci_high <- cellrows$log_rt+half
  cellrows$geometric_rt_ms <- exp(cellrows$log_rt)
  cellrows$geometric_ci_low_ms <- exp(cellrows$log_ci_low)
  cellrows$geometric_ci_high_ms <- exp(cellrows$log_ci_high)
  write.csv(cellrows,file.path(out,paste0(id,'_cell_estimates.csv')),row.names=FALSE)
  vectors <- list()
  vector <- function(terms,weights) {
    x <- setNames(rep(0,length(fixef(fit))),names(fixef(fit)))
    names_requested <- paste0('cell',terms)
    stopifnot(all(names_requested %in% names(x)))
    for (i in seq_along(terms)) x[names_requested[i]] <- x[names_requested[i]]+weights[i]
    x
  }
  key <- function(run,cond) sprintf('r%02d_%s',run,cond)
  for (k in 1:8) for (c in c('Str1','Str2')) {
    if (key(k,c) %in% cellrows$cell)
      vectors[[paste0('run',k,'_',c,'_minus_Ran')]] <- vector(c(key(k,c),key(k,'Ran')),c(1,-1))
  }
  vectors[['Str1_run2_minus_run1']] <- vector(c(key(2,'Str1'),key(1,'Str1')),c(1,-1))
  vectors[['Str2_run8_minus_run3']] <- vector(c(key(8,'Str2'),key(3,'Str2')),c(1,-1))
  for (k in 3:6) vectors[[paste0('run',k,'_Str2_minus_Str1')]] <- vector(c(key(k,'Str2'),key(k,'Str1')),c(1,-1))
  base <- vector(c(key(1,'Str1'),key(2,'Str1'),key(1,'Ran'),key(2,'Ran')),c(.5,.5,-.5,-.5))
  phases <- list()
  for (i in 1:3) {
    stage <- c('early','middle','late')[i];k <- i*2+1
    phases[[stage]] <- vector(c(key(k,'Str2'),key(k+1,'Str2'),key(k,'Ran'),key(k+1,'Ran')),c(.5,.5,-.5,-.5))-base
    vectors[[paste0(stage,'_adjusted_change')]] <- phases[[stage]]
  }
  for (pair in list(c('early','middle'),c('early','late'),c('middle','late')))
    vectors[[paste(pair,collapse='_minus_')]] <- phases[[pair[1]]]-phases[[pair[2]]]
  stopifnot(length(vectors)==24)
  contrasts <- lapply(names(vectors),function(name) {
    v <- vectors[[name]];estimate <- sum(v*fixef(fit));se <- sqrt(drop(t(v)%*%vc%*%v))
    df_nlme <- min(fit$fixDF$X[abs(v)>1e-12]);df <- min(df_nlme,nlevels(d$subject)-1);stat <- estimate/se
    half <- qt(.975,df)*se
    data.frame(policy=id,contrast=name,log_difference=estimate,se=se,df_nlme=df_nlme,df=df,t=stat,
      p_two_sided=2*pt(-abs(stat),df),log_ci_low=estimate-half,log_ci_high=estimate+half,
      geometric_rt_ratio=exp(estimate),ratio_ci_low=exp(estimate-half),ratio_ci_high=exp(estimate+half))
  })
  contrasts <- do.call(rbind,contrasts);contrasts$q_bh_24 <- p.adjust(contrasts$p_two_sided,'BH')
  write.csv(contrasts,file.path(out,paste0(id,'_contrasts.csv')),row.names=FALSE)
  write.csv(data.frame(contrast=names(vectors),do.call(rbind,vectors),check.names=FALSE),
     file.path(out,paste0(id,'_contrast_weights.csv')),row.names=FALSE)
  # Joint Wald test of departures from additive run and condition effects.
  additive <- model.matrix(~run+condition,data=transform(cellmeta,run=factor(session,levels=1:8)))
  X <- model.matrix(~0+cell,cellmeta);addcoef <- solve(X,additive)
  qrfull <- qr.Q(qr(addcoef),complete=TRUE)
  K <- t(qrfull[,seq.int(qr(addcoef)$rank+1,ncol(qrfull)),drop=FALSE])
  dimnames(K) <- list(NULL,names(fixef(fit)))
  est <- drop(K%*%fixef(fit));covK <- K%*%vc%*%t(K)
  numerator_df <- qr(K)$rank
  Fvalue <- drop(t(est)%*%solve(covK,est))/numerator_df
  denominator_df <- min(nlevels(d$subject)-1,min(fit$fixDF$X))
  omnibus <- data.frame(test='Run-by-condition departures from additive fixed cell means',
    numerator_df=numerator_df,denominator_df=denominator_df,F=Fvalue,
    p=pf(Fvalue,numerator_df,denominator_df,lower.tail=FALSE),
    method='Joint Wald F with conservative participant DF cap')
  write.csv(omnibus,file.path(out,paste0(id,'_run_condition_wald.csv')),row.names=FALSE)
  residual <- residuals(fit,type='normalized')
  diagnostics <- data.frame(subject=d$subject,run=d$session,trial=d$trial_in_run,
                              residual=residual,fitted=fitted(fit))
  acf <- lapply(split(diagnostics,interaction(diagnostics$subject,diagnostics$run)),function(a) {
    a <- a[order(a$trial),];keep <- diff(a$trial)==1
    r <- if(sum(keep)>2) cor(head(a$residual,-1)[keep],tail(a$residual,-1)[keep])else NA_real_
    data.frame(subject=as.character(a$subject[1]),run=a$run[1],lag1=r)
  })
  acf <- do.call(rbind,acf);write.csv(acf,file.path(out,paste0(id,'_residual_lag1.csv')),row.names=FALSE)
  pdf(file.path(out,paste0(id,'_diagnostics.pdf')),width=10,height=5)
  par(mfrow=c(1,2));plot(fitted(fit),residual,pch=16,cex=.1,
    xlab='Fitted log RT (ms)',ylab='Normalized residual');abline(h=0,col='gray')
  qqnorm(residual,pch=16,cex=.15);qqline(residual,col='red');dev.off()
  status <- list(status='completed',policy=id,n_trials=nrow(d),n_subjects=nlevels(d$subject),
    n_observed_cells=nlevels(d$cell),design_rank=qr(model.matrix(~0+cell,d))$rank,
    warnings=warnings,AR1=as.numeric(coef(fit$modelStruct$corStruct,unconstrained=FALSE)),
    mean_normalized_residual_lag1=mean(acf$lag1,na.rm=TRUE),
    residual_sigma=fit$sigma,condition_SD_ratios=as.list(coef(fit$modelStruct$varStruct,unconstrained=FALSE)),
    log_likelihood=as.numeric(logLik(fit)),variance_components=capture.output(VarCorr(fit)),
    n_contrasts=24,n_fdr_significant=sum(contrasts$q_bh_24<=.05))
  write_json(status,file.path(out,paste0(id,'_status.json')),pretty=TRUE,auto_unbox=TRUE)
  print(status);print(contrasts)
  invisible(fit)
}
primary <- fit_one(raw,'primary')
sensitivity <- fit_one(raw[raw$p_response_rt>=.1,],'min100')
writeLines(capture.output(sessionInfo()),file.path(out,'R_session_info.txt'))
