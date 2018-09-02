# Preparing data for Universal Language Model Fine-Tuning for Text Classification
library(text2vec)
library(caret)
library(feather)
library(keras)
library(parallel)
cl <- makeCluster(16)
library(readr)
library(hunspell)
set.seed(78910)

setwd("/app/project/txtm_news/Unclassified/Trade/Shipping Stat Commodity Coding/Neural Network/CNN/ULMFiT_July2018/Trial2")
CC9c <- read_feather("/app/project/txtm_news/Unclassified/Trade/Shipping Stat Commodity Coding/Neural Network/CNN/ULMFiT_July2018/Trial2/CC9c_Aug3.feather")
correct <- read_csv("/app/project/txtm_news/Unclassified/Trade/Shipping Stat Commodity Coding/Raw Data/Doubtful Classification_Aug8/Classification_Combined.csv")

# 67,969 EN tokens
vocab_EN10 <- create_vocabulary(itoken(EN$Des)) %>% prune_vocabulary(doc_proportion_max=0.1, doc_count_min=10L)

# 2,217 CN character level tokens
vocab_CN <- create_vocabulary(itoken(onlyCN$Des)) %>% prune_vocabulary(doc_proportion_max=0.1)

# Combining EN10 and CN
vocab <- rbind(vocab_EN10, vocab_CN)

# Adding token for 'unknown', 'padding' and 'beginning of sentence' token
Nv <- length(vocab$term)
vocab[Nv+1,]$term <- "_unk_"
vocab[Nv+1,]$term_count <- max(vocab$term_count) + 3
vocab[Nv+2,]$term <- "_pad_"
vocab[Nv+2,]$term_count <- max(vocab$term_count) + 2
vocab[Nv+3,]$term <- "_bos_"
vocab[Nv+3,]$term_count <- max(vocab$term_count) + 1

# Sort vocab by term_count, using id for less confusion in later stage involving python
vocab <- vocab[order(vocab$term_count, decreasing = T),]
vocab$id <- 0:(nrow(vocab)-1)
write_feather(vocab, "vocab_EN10_CN.feather")

# Correcting wrong labels (human mistakes)
CC9c$new <- correct$Corrected[match(CC9c$RefNo, correct$RefNo)]
CC9c$SubCode[!is.na(CC9c$new)] <- CC9c$new[!is.na(CC9c$new)]

# Prepare the tokeniser
it_k <- text_tokenizer(num_words = nrow(vocab), split = " ", filters='', char_level = FALSE, lower=TRUE)
word_index <- setNames(as.list(vocab$id), vocab$term)
it_k$word_index <- word_index

# train:valid:test = 3:2:1 for classifier training
inGP <- createFolds(CC9c$SubCode, k = 6, list = FALSE, returnTrain = FALSE)
trainid <- inGP %in% c(2, 3, 4)
validid <- inGP %in% c(1, 5)
testid <- inGP %in% c(6)

CC9c$group <- NA
CC9c$group[trainid] <- "train"
CC9c$group[validid] <- "valid"
CC9c$group[testid] <- "test"

# Saving target variables
write_feather(as.data.frame(CC9c$SubCode[trainid]), "lbl_trn_trial3.feather")
write_feather(as.data.frame(CC9c$SubCode[validid]), "lbl_val_trial3.feather")
write_feather(as.data.frame(CC9c$SubCode[testid]), "lbl_test_trial3.feather")

# 1 is for "_pad_" in the vocab
t2s <- texts_to_sequences(it_k, CC9c$Des)
# reversing the elements ("office equipment" --> "equipment office")
t2s_bwd <- parLapply(cl, t2s, rev)

# train:valid = 9:1 for tuning the LM
inLM <- createFolds(CC9c$SubCode, k = 10, list = F, returnTrain = F)

# token indices to python for LM tuning
t2s_trn_lm <- t2s[inLM %in% c(1:9)]
t2s_val_lm <- t2s[inLM %in% c(10)]
t2s_bwd_trn_lm <- t2s_bwd[inLM %in% c(2:10)]
t2s_bwd_val_lm <- t2s_bwd[inLM %in% c(1)]

write_feather(as.data.frame(unlist(t2s_trn_lm)), "trn_lm_trial3.feather")
write_feather(as.data.frame(unlist(t2s_val_lm)), "val_lm_trial3.feather")

write_feather(as.data.frame(unlist(t2s_bwd_trn_lm)), "trn_lm_trial3_bwd.feather")
write_feather(as.data.frame(unlist(t2s_bwd_val_lm)), "val_lm_trial3_bwd.feather")

# Remove the beginnis of sentence tag "_bos_" for classifier input
CC9c$Des <- parSapply(cl, CC9c$Des, rm_bos)

t2s <- texts_to_sequences(it_k, CC9c$Des)
t2s_bwd <- parLapply(cl, t2s, rev)

# max. 298 tokens after padding
t2s_df <- as.data.frame(pad_sequences(t2s, padding="post", value=1))
t2s_bwd_df <- as.data.frame(pad_sequences(t2s_bwd, padding="post", value=1))

# padded sequences as input
write_feather(t2s_df[trainid,], "seq_trn_trial3.feather")
write_feather(t2s_df[validid,], "seq_val_trial3.feather")
write_feather(t2s_df[testid,], "seq_test_trial3.feather")

write_feather(t2s_bwd_df[trainid,], "seq_trn_trial3_bwd.feather")
write_feather(t2s_bwd_df[validid,], "seq_val_trial3_bwd.feather")
write_feather(t2s_bwd_df[testid,], "seq_test_trial3_bwd.feather")

stopCluster(cl)