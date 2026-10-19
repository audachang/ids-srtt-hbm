#!/usr/bin/env Rscript
# Replay recovered source cleaning/learner tests without executing legacy I/O.
suppressPackageStartupMessages(library(dplyr))
suppressPackageStartupMessages(library(trimr))
suppressPackageStartupMessages(library(jsonlite))
args <- commandArgs(trailingOnly=TRUE)
stopifnot(length(args)==2)
root <- normalizePath(args[1],mustWork=TRUE); out <- args[2]
dir.create(out,recursive=TRUE,showWarnings=FALSE)
archive <- '/media/data2/pinwei/Joanne_SRT_2/For_teamNCU_Joanne'
helper <- file.path(archive,'My_modifiedRecursive.R')
source(helper)
subjects20 <- readLines(file.path(root,'code/replication/config/subjects.txt'))
subjects22 <- c(subjects20,'sub-005','sub-023')
files <- sort(list.files(file.path(archive,'Import_data'),pattern='[.]csv$',full.names=TRUE))
stopifnot(length(files)==176)
frames <- lapply(files,function(path) {
 d <- read.csv(path,stringsAsFactors=FALSE)
 d <- d[!is.na(d$keypress),]
 stopifnot(nrow(d)==320,length(unique(d$participant))==1,length(unique(d$session))==1)
 p <- as.integer(d$participant[1]); run <- as.integer(d$session[1])
 canonical <- read.csv(file.path(root,'data/behavioral',sprintf('%03d_run_%d_srtt_prob_new.csv',p,run)),stringsAsFactors=FALSE)
 stopifnot(nrow(canonical)==320)
 for (column in c('block','rule','p_response_keys','rule_group')) canonical[[column]] <- as.character(canonical[[column]])
 first <- as.character(d$Stimuli_key_resp.keys);second <- as.character(d$Stimuli_key_resp_2.keys)
 response <- ifelse(seq_len(nrow(d))%%2==1,first,second)
 rt <- case_when(first=='None' & second=='None' ~ NA_real_,
                 first!='None' & second=='None' ~ d$Stimuli_key_resp.rt,
                 first=='None' & second!='None' ~ d$Stimuli_key_resp_2.rt)
 correct <- ifelse(response==d$keypress,1L,0L)
 stopifnot(isTRUE(all.equal(rt,canonical$p_response_rt,tolerance=1e-10,check.attributes=FALSE)),
           all(ifelse(is.na(correct),0,correct)==canonical$correct.resp))
 canonical$subject <- sprintf('sub-%03d',p)
 canonical$trial_in_run <- seq_len(nrow(canonical))
 canonical$correct_response <- as.integer(canonical$correct.resp==1 & is.finite(canonical$p_response_rt))
 canonical$omitted <- !is.finite(canonical$p_response_rt)|canonical$p_response_keys %in% c('None','nan','')
 canonical$source_file <- path
 canonical
})
raw <- bind_rows(frames) %>% arrange(participant,session,trial_in_run)
stopifnot(nrow(raw)==56320,setequal(unique(raw$subject),subjects22),all(table(raw$subject,raw$session)==320))
write.csv(raw,file.path(out,'raw_trials_N22.csv'),row.names=FALSE)
write.csv(data.frame(path=files),file.path(out,'native_input_files.csv'),row.names=FALSE)
writeLines(capture.output(sessionInfo()),file.path(out,'R_session_info.txt'))
# Original helper and call, with original block labels and no added RT cutoff.
clean <- My_modifiedRecursive(data=as.data.frame(raw),minRT=0,condVar='block',
                             accVar='correct.resp',rtVar='p_response_rt',returnType='raw')
clean <- clean[order(clean$participant,clean$session,clean$trial_in_run),]
stopifnot(all(clean$correct.resp==1),all(is.finite(clean$p_response_rt)),all(clean$p_response_rt>0),
          !anyDuplicated(clean[c('subject','session','trial_in_run')]))
write.csv(clean,file.path(out,'custom_cleaned_trials_N22.csv'),row.names=FALSE)
stock <- trimr::modifiedRecursive(data=as.data.frame(raw[is.finite(raw$p_response_rt)&raw$p_response_rt>0&raw$p_response_rt<1,]),
 minRT=0,pptVar='participant',condVar='block',rtVar='p_response_rt',accVar='correct_response',omitErrors=TRUE,returnType='raw')
key <- function(d) paste(d$subject,d$session,d$trial_in_run,sep='/')
comparison <- data.frame(trial_key=union(key(clean),key(stock)))
comparison$custom_retained <- comparison$trial_key %in% key(clean)
comparison$stock_block_retained <- comparison$trial_key %in% key(stock)
write.csv(comparison,file.path(out,'custom_vs_stock_block_cleaning.csv'),row.names=FALSE)
# Replay the recovered learner-test specification; final-block test is separate.
learning <- list()
for (p in sort(unique(clean$participant))) for (run in c(2,8)) for (scope in c('recovered_all_six_structured','last_structured_sensitivity')) {
 blocks <- c(paste0('Ses',run,'Str_',if(scope=='recovered_all_six_structured')1:6 else 6),paste0('Ses',run,'Ran2'))
 d <- clean[clean$participant==p & clean$session==run & clean$block %in% blocks,]
 d$group <- ifelse(d$rule=='Ran','Ran','Str');d$rt_ms <- d$p_response_rt*1000
 stopifnot(length(unique(d$group))==2)
 f <- var.test(rt_ms~group,d);equal <- f$p.value>.05
 tt <- t.test(rt_ms~group,d,var.equal=equal,alternative='greater')
 two <- t.test(rt_ms~group,d,var.equal=equal,alternative='two.sided')
 learning[[length(learning)+1]] <- data.frame(subject=sprintf('sub-%03d',p),participant=p,run=run,scope=scope,
  n_random=sum(d$group=='Ran'),n_structured=sum(d$group=='Str'),mean_random_ms=mean(d$rt_ms[d$group=='Ran']),
  mean_structured_ms=mean(d$rt_ms[d$group=='Str']),benefit_ms=unname(diff(rev(tt$estimate))),
  f_test_p=f$p.value,equal_variances=equal,t=unname(tt$statistic),df=unname(tt$parameter),
  p_one_sided=tt$p.value,p_two_sided=two$p.value,ci_low_ms=two$conf.int[1],ci_high_ms=two$conf.int[2],
  meets_p005=tt$p.value<.05)
}
learning <- bind_rows(learning)
write.csv(learning,file.path(out,'recovered_learner_test.csv'),row.names=FALSE)
# Cohort-specific trial tables and descriptive rates use the same cleaning.
summary <- list()
for (cohort in c('N20','N22')) {
 ids <- if(cohort=='N20')subjects20 else subjects22
 folder <- file.path(out,cohort);dir.create(folder,recursive=TRUE,showWarnings=FALSE)
 r <- raw[raw$subject %in% ids,];c <- clean[clean$subject %in% ids,]
 write.csv(c,file.path(folder,'cleaned_trials.csv'),row.names=FALSE)
 write.csv(r,file.path(folder,'raw_trials.csv'),row.names=FALSE)
 rates <- r %>% group_by(subject,session,block,rule_group) %>%
  summarise(n_trials=n(),n_errors=sum(correct_response!=1),n_correct=sum(correct_response==1),
            n_anticipatory=sum(is.finite(p_response_rt)&p_response_rt>0&p_response_rt<.1&!omitted),.groups='drop')
 rates$error_rate <- rates$n_errors/rates$n_trials
 write.csv(rates,file.path(folder,'participant_block_error_rates.csv'),row.names=FALSE)
 summary[[cohort]] <- data.frame(cohort=cohort,n_subjects=length(ids),n_raw=nrow(r),n_custom_clean=nrow(c),
  n_min100=sum(c$p_response_rt>=.1),n_errors=sum(r$correct_response!=1),n_omitted=sum(r$omitted),
  n_anticipatory=sum(is.finite(r$p_response_rt)&r$p_response_rt>0&r$p_response_rt<.1&!r$omitted))
}
write.csv(bind_rows(summary),file.path(out,'preprocessing_summary.csv'),row.names=FALSE)
write_json(list(passed=TRUE,n_native_files=176,n_trials_native_rt_verified=56320,
 n_custom_clean=nrow(clean),custom_vs_stock_block_trial_differences=sum(comparison$custom_retained!=comparison$stock_block_retained),
 recovered_run2_nonlearners=learning$subject[learning$run==2 & learning$scope=='recovered_all_six_structured' & !learning$meets_p005],
 original_historical_decision_log_recovered=FALSE),file.path(out,'replay_status.json'),pretty=TRUE,auto_unbox=TRUE)
print(bind_rows(summary));print(learning[learning$run==2 & learning$scope=='recovered_all_six_structured',c('subject','benefit_ms','p_one_sided','meets_p005')])
