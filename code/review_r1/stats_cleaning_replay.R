#!/usr/bin/env Rscript
# Reproduce existing cleaning only; never refit a model or run legacy source I/O.
suppressPackageStartupMessages(library(dplyr))
suppressPackageStartupMessages(library(trimr))
suppressPackageStartupMessages(library(jsonlite))
args <- commandArgs(trailingOnly=TRUE)
stopifnot(length(args)==2)
root <- normalizePath(args[1],mustWork=TRUE)
out <- normalizePath(args[2],mustWork=TRUE)
stopifnot(startsWith(out,paste0(root,'/results/behavioral/review_r1_20261019/')))
raw <- read.csv(file.path(root,'results/behavioral/reanalysis_20261018_updated_n22/source_replay/raw_trials_N22.csv'),stringsAsFactors=FALSE)
saved <- read.csv(file.path(root,'results/behavioral/reanalysis_20261018_updated_n22/source_replay/N22/cleaned_trials.csv'),stringsAsFactors=FALSE)
source('/media/data2/pinwei/Joanne_SRT_2/For_teamNCU_Joanne/My_modifiedRecursive.R')
clean <- My_modifiedRecursive(data=raw,minRT=0,condVar='block',accVar='correct.resp',
                             rtVar='p_response_rt',returnType='raw')
sortkeys <- function(d) d[order(d$subject,d$session,d$trial_in_run),]
clean <- sortkeys(clean); saved <- sortkeys(saved)
cols <- c('subject','session','trial_in_run','rule_group','p_response_rt')
stopifnot(isTRUE(all.equal(clean[,cols],saved[,cols],tolerance=0,check.attributes=FALSE)),
          nrow(clean)==53959,all(clean$correct.resp==1),all(clean$p_response_rt>0),
          all(clean$p_response_rt<1),sum(clean$p_response_rt>=.1)==52126)
write_json(list(passed=TRUE,R_version=R.version.string,trimr_version=as.character(packageVersion('trimr')),
  retained_trial_keys_and_RT_exact=TRUE,n_primary=nrow(clean),n_min100=sum(clean$p_response_rt>=.1),
  no_LME_refit=TRUE,method='Original custom block-wise helper, minRT=0; min100 applied after cleaning'),
  file.path(out,'cleaning_replay_validation.json'),pretty=TRUE,auto_unbox=TRUE)
writeLines(capture.output(sessionInfo()),file.path(out,'R_session_info.txt'))
cat('CLEANING REPLAY PASS:',nrow(clean),'primary;',sum(clean$p_response_rt>=.1),'min100\n')
