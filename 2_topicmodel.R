# install.packages("topicmodels")
# install.packages("quanteda")
# install.packages("tidyverse")
# install.packages("tidytext")
# install.packages("lubridate")
# install.packages("sysfonts")
# install.packages("showtext")
# install.packages("jiebaR")
# install.packages("servr")
# install.packages("ldatuning")
# install.packages("doParallel")
# install.packages("reshape2")

library(topicmodels)
library(quanteda)
library(tidyverse)
library(lubridate)
library(sysfonts)
font_add_google("Noto Sans TC", "Noto Sans TC")
library(showtext)
showtext_auto()
library(jiebaR)
library(tidytext)

####################################################################################
## Creating DFMs                                                                  ##
####################################################################################

df_tsai <-  read.csv("tsai.csv")
corpus_tsai <- corpus(df_tsai, text_field = "Message")

tokeniser <- worker()
raw_texts <- as.character(corpus_tsai)
tokenised_texts <- purrr::map(raw_texts, segment, tokeniser)
tokens_tsai <- tokens(tokenised_texts,
                     remove_punct = TRUE, 
                     remove_numbers = TRUE, 
                     remove_url = TRUE,
                     remove_symbols = TRUE,
                     verbose = TRUE)
dfm_tsai <- dfm(tokens_tsai)
customstopwords <- c()
dfm_tsai <- dfm_remove(dfm_tsai, c(stopwords('chinese', source = "misc"), stopwords('english'), customstopwords))

topfeatures(dfm_tsai, 30) 

# Trimming DFM to reduce training time
docvars(dfm_tsai, "docname") <- docnames(dfm_tsai)
dfm_trimmed <- dfm_trim(dfm_tsai, min_docfreq = 5, min_count = 10)
dfm_trimmed

# Removing rows that contain all zeros
row_sum <- apply(dfm_trimmed , 1, sum)
dfm_trimmed <- dfm_trimmed[row_sum> 0, ]

# Converting to another format
lda_data <- convert(dfm_trimmed, to = "topicmodels")
lda_data

####################################################################################
## Finding K                                                                      ##
####################################################################################

######################### LDA Tuning ######################### 
library("ldatuning")

ldatuning.result <- FindTopicsNumber(
  lda_data,
  topics = seq(from = 10, to = 50, by = 10),
  metrics = c("Griffiths2004", "CaoJuan2009", "Arun2010"), # There are 4 possible metrics: Griffiths2004", "CaoJuan2009", "Arun2010", "Deveaud2014"
  method = "Gibbs",
  control = list(seed = 4321),
  verbose = TRUE
)
FindTopicsNumber_plot(ldatuning.result)


######################### Perplexity ######################### 

# Another approach to find K is perplexity: we split the data into 5 parts, 
# train a model using 4 of the 5 parts, then see how well does it predict the held-out one, repeat 5 times.
# Note that this approach can take a long time since we need to train 5 models for each candidate K

library(doParallel)
library(dplyr)
library(reshape2)
library(tidyr)
library(ggplot2)

# Here we do parallelisation to speed up the process.
cluster <- makeCluster(detectCores(logical = TRUE)-1, outfile = "Log.txt")
registerDoParallel(cluster)
clusterEvalQ(cluster, {
  library(topicmodels)
})

n <- nrow(lda_data)
burnin <- 1000
iter <- 1000
keep <- 50
folds <- 5
splitfolds <- sample(1:folds, n, replace = TRUE)
candidate_k <- c(10, 20, 30, 40, 50) # candidates for how many topics

clusterExport(cluster, c("lda_data", "burnin", "iter", "keep", "splitfolds", "folds", "candidate_k"))

system.time({
  results <- foreach(j = 1:length(candidate_k), .combine = rbind) %dopar%{
    k <- candidate_k[j]
    results_1k <- matrix(0, nrow = folds, ncol = 2)
    colnames(results_1k) <- c("k", "perplexity")
    for(i in 1:folds){
      train_set <- lda_data[splitfolds != i , ]
      valid_set <- lda_data[splitfolds == i, ]
      
      fitted <- LDA(train_set, k = k, method = "Gibbs",
                    control = list(burnin = burnin, iter = iter, keep = keep) )
      results_1k[i,] <- c(k, perplexity(fitted, newdata = valid_set))
    }
    print(k)
    return(results_1k)
  }
})
stopCluster(cluster)

# Plotting the results
results_df <- as.data.frame(results)
results_df$istest <- "test"
avg_perplexity <- results_df %>% group_by(k) %>% summarise(perplexity = mean(perplexity))
avg_perplexity$istest <- "avg"
plot_df <- rbind(results_df, avg_perplexity)

ggplot(plot_df, aes(x = k, y = perplexity, group = istest)) +
  geom_point(aes(colour = factor(istest))) +
  geom_line(data = subset(plot_df, istest %in% "avg"), color = "red") +
  ggtitle("5-fold Cross-validation of Topic Modelling") +
  labs(x = "Candidate k", y = "Perplexity") +
  scale_x_discrete(limits = candidate_k) +
  scale_color_discrete(name="Test\\Average",
                       breaks=c("test", "avg"),
                       labels=c("Test", "Average")) +
  theme_bw() +
  theme(plot.title = element_text(hjust = 0.5))


####################################################################################
## Fit Optimal Model(s)                                                           ##
####################################################################################

# A good practice would be to fit multiple models from our K search and compare their performance
# But here we are fitting a toy model to save time

lda_model <- LDA(lda_data, 10, method="Gibbs")  

get_terms(lda_model, k=20)


####################################################################################
##  Visualization                                                                 ##
####################################################################################

######################### Visualize Terms ##############################
library(tidytext)
library(tidyr)

# First we need to extract the beta (probability of a word in a topic)
topics <- tidy(lda_model, matrix = "beta")

topics %>%
  group_by(topic) %>%
  top_n(10, beta) %>%
  ungroup() %>%
  arrange(topic, -beta) %>% 
  mutate(term = reorder_within(term, beta, topic)) %>%
  ggplot(aes(beta, term, fill = factor(topic))) +
  geom_col(show.legend = FALSE) +
  facet_wrap(~ topic, scales = "free") +
  scale_y_reordered()

######################### LDAvis ##############################
library(topicmodels)
library(dplyr)
library(stringi)
library(quanteda)
library(LDAvis)

# LDAvis is an interactive tool to visualise a LDA model
# It is useful for initial exploration, especially the relationship between topics

# There might be a discrepancy between the dfm we feed the function,
# and the dfm actually used for training (some rows might be excluded)
visdfm <- dfm_subset(dfm_trimmed, docname %in% rownames(lda_data))

# A custom function to transform data to json format
topicmodels_json_ldavis <- function(fitted, dfm, dtm){
  # Find required quantities
  phi <- posterior(fitted)$terms %>% as.matrix
  theta <- posterior(fitted)$topics %>% as.matrix
  vocab <- colnames(phi)
  doc_length <- ntoken(dfm)
  
  temp_frequency <- as.matrix(dtm)
  freq_matrix <- data.frame(ST = colnames(temp_frequency),
                            Freq = colSums(temp_frequency))
  rm(temp_frequency)
  # Convert to json
  json_lda <- LDAvis::createJSON(phi = phi, theta = theta,
                                 vocab = vocab,
                                 doc.length = doc_length,
                                 term.frequency = freq_matrix$Freq)
  return(json_lda)
}

json_lda <- topicmodels_json_ldavis(lda_model, visdfm, lda_data)
serVis(json_lda, out.dir = "LDAvis", open.browser = TRUE)


######################### Topic Proportion per Document ##############################
doc_gamma <- tidy(lda_model, matrix = "gamma")

doc_gamma %>%
  filter(document %in% c("text1","text2","text3","text4","text5","text6")) %>% 
  ggplot(aes(factor(topic), gamma, fill = factor(topic))) +
  geom_col() +
  facet_wrap(~ document) +
  labs(x = "topic", y = expression(gamma))



####################################################################################
##  Validation                                                                    ##
####################################################################################

# Before we do anything, it is a good idea to have a data frame that contains
# all document-level information and the topic for each document
topic_df <- docvars(corpus_tsai)
topic_df$doc_name <- docnames(corpus_tsai)
topic_df$text <- as.character(corpus_tsai)

lda_df <- data.frame(topic = get_topics(lda_model), 
                     doc_name = lda_model@documents)
topic_df <- left_join(topic_df, lda_df, by = "doc_name")
topic_df$date <- as.Date(topic_df$Post.Created.Date)
topic_df$topic <- as.factor(topic_df$topic)


######################### Accuracy per Topic ##############################

# Accuracy, this is the bare minimum that we should do
validation_df <- topic_df %>% 
  group_by(topic) %>% 
  slice_sample(n = 10, replace = TRUE)

# write.csv(validation_df, "topic_validation_random.csv")


# Get Top Texts of each Topics
doc_gamma <- tidy(lda_model, matrix = "gamma")
gamma_df <- doc_gamma %>% group_by(topic) %>% arrange(desc(gamma)) %>% top_n(10) %>% arrange(topic)
gamma_df <- left_join(gamma_df, select(topic_df, doc_name, text, URL), by = c("document" = "doc_name"))

#write.csv(gamma_df, "topic_validation_gamma.csv")

######################### Word Intrusion Test ##############################

# Note some OS might fail to install this package, 
# install.packages("oolong")
library(oolong)

oolong_test <- wi(lda_model)
oolong_test$do_word_intrusion_test()
oolong_test$lock(force = TRUE)
oolong_test

######################### Topic Intrusion Test ##############################
oolong_test <- ti(lda_model, corpus_tsai)
oolong_test$do_topic_intrusion_test()
oolong_test$lock(force = TRUE)
oolong_test

####################################################################################
##  Visualize Topics                                                              ##
####################################################################################

# We can see counts
table(topic_df$topic)

topic_df %>% 
  group_by(topic) %>% 
  count() %>% 
  ggplot(aes(topic, n, fill = topic)) +
  geom_col() +
  coord_flip()


# Or even trend over time
topic_df %>% 
  mutate(date = floor_date(date, "week")) %>% 
  group_by(date, topic) %>% 
  count() %>% 
  ggplot(aes(date, n, color = topic)) +
  geom_line() +
  facet_wrap(vars(topic))

####################################################################################
## Topic Network                                                                  ##
####################################################################################

# In some cases, it might be useful to visualize a topic model as a network
# A use case can be found: https://doi.org/10.1111/nana.12805

# install.package("igraph")
# install.package("ggnewscale")
# install.package("ggnetwork")
# install.package("RColorBrewer")

library(igraph)
library(ggnewscale)
library(ggnetwork)
library(RColorBrewer)

beta_matrix <- lda_model@beta
beta_matrix <- t(beta_matrix) # Transform it so that col: topic, row: word

# Correlation between columns (word usage)
cor_matrix <- cor(beta_matrix, method = "pearson")
colnames(cor_matrix) <- 1:10
rownames(cor_matrix) <- 1:10
diag(cor_matrix) <- 0
quantiles <- quantile(as.vector(cor_matrix), c(.8, .9, .95, .99))

# Turn it into a network
cor_network <- graph_from_adjacency_matrix(cor_matrix, mode = "upper", weighted = TRUE)
cor_network <- igraph::delete.edges(cor_network, which(E(cor_network)$weight <= quantiles[1])) # removing all edges below 80%

V(cor_network)$degree <- igraph::degree(cor_network) # Here we use degree centrality, but we can use other measures instead

# Splitting edges into 4 sub-datasets for visualisation
top99 <- function(x) { x[ x$weight >= quantiles[4], ] }
top95 <- function(x) { x[ x$weight >= quantiles[3] & x$weight < quantiles[4], ] }
top90 <- function(x) { x[ x$weight >= quantiles[2] & x$weight < quantiles[3], ] }
top80 <- function(x) { x[ x$weight < quantiles[2], ] }

ggplot(ggnetwork(cor_network), aes(x = x, y = y, xend = xend, yend = yend)) +
  geom_edges(aes(size = 0.5),  color = brewer.pal(n = 5, name = "Greys")[2], 
             curvature = 0.15, alpha = 0.4, show.legend = FALSE,
             data = top80) +
  geom_edges(aes(size = 0.75),  color = brewer.pal(n = 5, name = "Greys")[3], 
             curvature = 0.15, alpha = 0.4, show.legend = FALSE,
             data = top90) +
  geom_edges(aes(size = 1), color = "brown4",
             curvature = 0.15, alpha = 0.4, show.legend = FALSE,
             data = top95) +
  geom_edges(aes(size = 1), color = "brown2",
             curvature = 0.15, alpha = 0.7, show.legend = FALSE,
             data = top99) +
  new_scale_color() +
  geom_nodes(aes(x, y, size = (degree + 1)), alpha = 0.9)+
  scale_size_area("degree", max_size = 20) +
  geom_nodelabel_repel(aes(label = name), size = 6, alpha = 0.8, segment.size = 5) +
  theme_minimal() +
  theme(axis.text = element_blank(),
        axis.title = element_blank(),
        panel.background = element_blank(),
        panel.grid = element_blank(),
        legend.text=element_text(size=16),
        legend.title=element_text(size=16),
        legend.position = "bottom",
        legend.background = element_rect(fill="white", size=0.5, linetype="solid")) +
  guides(size=FALSE, alpha=FALSE, fill=FALSE)



