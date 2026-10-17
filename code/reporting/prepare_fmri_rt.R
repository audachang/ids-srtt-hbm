#!/usr/bin/env Rscript
# Reconstruct the original Exp3 cycle plot for the exact imaging cohort.
suppressPackageStartupMessages(library(dplyr))
suppressPackageStartupMessages(library(trimr))
args <- commandArgs(trailingOnly=TRUE)
if (length(args) != 2) stop('Usage: prepare_fmri_rt.R SOURCE_ROOT OUTPUT_DIR')
root <- normalizePath(args[1], mustWork=TRUE)
out <- args[2]
dir.create(out, recursive=TRUE, showWarnings=FALSE)
subjects <- readLines(file.path(root, 'code/replication/config/subjects.txt'))
subjects <- subjects[nzchar(subjects)]
stopifnot(length(subjects)==20, !anyDuplicated(subjects))
frames <- list()
paths <- character()
for (s in subjects) for (k in 1:8) {
  id <- as.integer(sub('sub-', '', s))
  p <- file.path(root, 'data/behavioral', sprintf('%03d_run_%d_srtt_prob_new.csv', id, k))
  d <- read.csv(p, fileEncoding='UTF-8-BOM', stringsAsFactors=FALSE)
  stopifnot(nrow(d)==320, all(d$participant==id), all(d$session==k))
  d <- d[,c('participant','session','block','rule','keypress','p_response_keys',
             'p_response_rt','correct.resp','rule_group','stim_onset')]
  for (column in c('block','rule','p_response_keys','rule_group'))
    d[[column]] <- as.character(d[[column]])
  d$subject <- s
  d$trial_in_run <- seq_len(nrow(d))
  frames[[length(frames)+1]] <- d
  paths <- c(paths,p)
}
raw <- bind_rows(frames) %>% arrange(participant, session, trial_in_run)
stopifnot(nrow(raw)==51200, n_distinct(raw$subject)==20)
# Validate counterbalancing independently from the saved rule_group field.
expected <- ifelse(raw$rule=='Ran','Ran',ifelse(
  (raw$participant %% 2==1 & raw$rule=='StrA') |
  (raw$participant %% 2==0 & raw$rule=='StrB'),'Str1','Str2'))
stopifnot(all(expected==raw$rule_group))
# Assign repetitions before excluding trials, so errors do not shift cycles.
raw <- raw %>% group_by(subject, rule_group) %>%
  mutate(structure_cycle=(row_number()-1L)%/%8L+1L) %>% ungroup() %>%
  mutate(displayed_cycle=structure_cycle+ifelse(rule_group=='Str2',60L,0L),
         valid_rt=is.finite(p_response_rt)&p_response_rt>0&p_response_rt<1,
         omitted=!is.finite(p_response_rt)|p_response_keys %in% c('None','nan',''),
         correct_response=as.integer(correct.resp==1 & !omitted))
stopifnot(all(table(raw$subject,raw$rule_group)[,'Str1']==960),
          all(table(raw$subject,raw$rule_group)[,'Str2']==960))
valid <- as.data.frame(raw %>% filter(valid_rt))
clean <- trimr::modifiedRecursive(valid,minRT=0,pptVar='participant',condVar='block',
  rtVar='p_response_rt',accVar='correct_response',omitErrors=TRUE,returnType='raw')
stopifnot(all(clean$correct_response==1),all(clean$p_response_rt>0),
          all(clean$p_response_rt<1),n_distinct(clean$subject)==20)
cycles <- clean %>% filter(rule_group!='Ran') %>%
  group_by(subject,participant,session,rule_group,structure_cycle,displayed_cycle) %>%
  summarise(rt_ms=mean(p_response_rt)*1000,n_trials=n(),.groups='drop') %>%
  mutate(stage=ifelse(structure_cycle<=60,'Early','Late'))
stopifnot(!anyDuplicated(cycles[c('subject','rule_group','structure_cycle')]),
          all(cycles$n_trials>=1&cycles$n_trials<=8))
write.csv(cycles,file.path(out,'participant_cycles.csv'),row.names=FALSE)
write.csv(clean,file.path(out,'cleaned_trials.csv'),row.names=FALSE)
write.csv(raw %>% group_by(subject,session,rule_group) %>%
  summarise(n_trials=n(),n_correct=sum(correct_response),n_omitted=sum(omitted),
            n_invalid_rt=sum(!valid_rt),accuracy=mean(correct_response),
            omission_rate=mean(omitted),.groups='drop'),
  file.path(out,'participant_accuracy.csv'),row.names=FALSE)
write.csv(clean %>% group_by(subject,session,rule_group) %>%
  summarise(rt_ms=mean(p_response_rt)*1000,n_trials=n(),.groups='drop'),
  file.path(out,'participant_run_rt.csv'),row.names=FALSE)
write.csv(clean %>% filter(p_response_rt>=.1) %>% group_by(subject,session,rule_group) %>%
  summarise(rt_ms=mean(p_response_rt)*1000,n_trials=n(),.groups='drop'),
  file.path(out,'participant_run_rt_min100.csv'),row.names=FALSE)
write.csv(data.frame(path=paths),file.path(out,'raw_input_files.csv'),row.names=FALSE)
write.csv(data.frame(n_subjects=20,n_runs=160,n_raw=nrow(raw),
  n_error_or_omitted=sum(raw$correct_response==0),n_omitted=sum(raw$omitted),
  n_invalid_rt=sum(!raw$valid_rt),n_after_trim=nrow(clean),
  n_correct_valid=sum(raw$correct_response==1&raw$valid_rt),
  n_correct_rt_below100=sum(raw$correct_response==1&raw$valid_rt&raw$p_response_rt<.1),
  trimr_version=as.character(packageVersion('trimr'))),
  file.path(out,'preprocessing_summary.csv'),row.names=FALSE)
writeLines(capture.output(sessionInfo()),file.path(out,'R_session_info.txt'))
