####################################################################################
## Setting Up                                                                     ##
####################################################################################
download.file("https://raw.githubusercontent.com/justinchuntingho/Academia-Sinica-Topic-Modeling/master/ukraine.csv", "ukraine.csv")
download.file("https://raw.githubusercontent.com/justinchuntingho/Academia-Sinica-Topic-Modeling/master/ios.csv", "ios.csv")
download.file("https://raw.githubusercontent.com/justinchuntingho/Academia-Sinica-Topic-Modeling/master/tsai.csv", "tsai.csv")
download.file("https://raw.githubusercontent.com/justinchuntingho/Academia-Sinica-Topic-Modeling/master/policy_agendas_english.lcd", "policy_agendas_english.lcd")

install.packages("tidyverse")
install.packages("quanteda")
install.packages("quanteda.textplots")
install.packages("quanteda.textstats")
install.packages("sysfonts")
install.packages("showtext")
install.packages("jiebaR")

library(tidyverse)
library(quanteda)
library(quanteda.textplots)
library(quanteda.textstats)

####################################################################################
## Basic Text Analysis                                                            ##
####################################################################################

# Dataset: Tweets from https://twitter.com/Ukraine

# Loading the documents
df_ukraine <-  read.csv("ukraine.csv")
corpus_ukraine <- corpus(df_ukraine, text_field = "text")

# The followings are not necessary steps, but it is always a good idea to view a portion of your data
corpus_ukraine[1:10] # print the first 10 documents
ndoc(corpus_ukraine) # Number of Documents
nchar(corpus_ukraine[1:10]) # Number of character for the first 10 documents
ntoken(corpus_ukraine[1:10]) # Number of tokens for the first 10 documents
ntoken(corpus_ukraine[1:10], remove_punct = TRUE) # Number of tokens for the first 10 documents after removing punctuation

# Meta-data
head(docvars(corpus_ukraine))

# Creating DFM
tokens_ukraine <- tokens(corpus_ukraine,
                     remove_punct = TRUE,
                     remove_numbers = TRUE,
                     remove_url = TRUE,
                     verbose = TRUE)
dfm_ukraine <- dfm(tokens_ukraine)

# Inspecting the results
topfeatures(dfm_ukraine, 30)

# What do they say about "russian"?
kwic(tokens_ukraine, "russian", 3)

# Plotting a histogram
features <- topfeatures(dfm_ukraine, 100)  # Putting the top 100 words into a new object
data.frame(list(term = names(features), frequency = unname(features))) %>% # Create a data.frame for ggplot
  ggplot(aes(x = reorder(term,-frequency), y = frequency)) + # Plotting with ggplot2
  geom_point() +
  theme_bw() +
  labs(x = "Term", y = "Frequency") +
  theme(axis.text.x=element_text(angle=90, hjust=1))

# Doing it again, removing stop words this time!

# Defining custom stopwords
customstopwords <- c("amp", "just", "make", "stopword")

dfm_ukraine <- dfm_remove(dfm_ukraine, c(stopwords('english'), customstopwords))

# Inspecting the results again
topfeatures(dfm_ukraine, 30)

# Top words again
features <- topfeatures(dfm_ukraine, 100)  # Putting the top 100 words into a new object
data.frame(list(term = names(features), frequency = unname(features))) %>% # Create a data.frame for ggplot
  ggplot(aes(x = reorder(term,-frequency), y = frequency)) + # Plotting with ggplot2
  geom_point() +
  theme_bw() +
  labs(x = "Term", y = "Frequency") +
  theme(axis.text.x=element_text(angle=90, hjust=1))

# Wordcloud
textplot_wordcloud(dfm_ukraine)


####################################################################################
## Keyword Analysis                                                               ##
####################################################################################

# Before and After the Invasion
tstat_key <- textstat_keyness(dfm_ukraine,
                              target = docvars(dfm_ukraine, "war"))

head(tstat_key)
tail(tstat_key)

textplot_keyness(tstat_key)


####################################################################################
## Lexicon Approach                                                               ##
####################################################################################

# Lexicon approach is a powerful approach to identify word-of-interest in your corpus
# However, the quality of your analysis depends heavily on the quality of your lexica
# It also depends on the topic of interest, some topics are inherently easier 
# to come up with a list of lexicon while others are more difficult to capture by default

#########################  Sentiment Analysis ######################### 
# One common use for lexicon approach is to conduct sentiment analysis
# You start by constructing/downloading a list of words that denote certain sentiments (eg positive/negative)

# We will use the Lexicoder Sentiment Dictionary (2015)
data_dictionary_LSD2015

# Passing the dictionary to the dictionary function, you can also define your own dictionary
sentiment <- tokens_lookup(tokens_ukraine, data_dictionary_LSD2015) %>%
  dfm()
sentiment <- convert(sentiment, to = "data.frame")
sentiment["sentiment"] <- sentiment$positive + sentiment$neg_negative - sentiment$negative - sentiment$neg_positive

# Merging the result with the original data set
df_wsentiment <- cbind(df_ukraine, sentiment)

# Basic Exploration: mean sentiment score
df_wsentiment %>%
  group_by(war) %>%
  summarise(sentiment = mean(sentiment))

# Sentiment trend over time
df_wsentiment$date <- as.Date(df_wsentiment$created_at)

plot_df <- df_wsentiment %>%
  group_by(date) %>%
  summarise(sentiment = mean(sentiment))
ggplot(plot_df, aes(date, sentiment)) +
  geom_line()

# Here we are going to use a function from this package to aggregate by week/month
# install.packages("lubridate")
library(lubridate)

plot_df <- df_wsentiment %>%
  mutate(date = floor_date(date, "week")) %>%
  group_by(date) %>%
  summarise(sentiment = mean(sentiment))
ggplot(plot_df, aes(date, sentiment)) +
  geom_line()


#########################  Policy Agendas ######################### 
# Lexicoder Policy Agenda dictionary captures major topics from the comparative Policy Agenda project 
# More information can be found here: http://www.snsoroka.com/data-lexicoder/

cap_dict <- dictionary(file = "policy_agendas_english.lcd")

policy <- tokens_lookup(tokens_ukraine, cap_dict) %>%
  dfm()
policy <- convert(policy, to = "data.frame")
df_wpolicy <- cbind(df_ukraine, policy)

# Basic Exploration
df_wpolicy %>%
  group_by(war) %>%
  select(war, macroeconomics:religion) %>% 
  summarise_all(sum) %>% 
  View()

# Some basic plotting
df_wpolicy %>%
  group_by(war) %>%
  select(war, macroeconomics:religion) %>% 
  summarise_all(sum) %>% 
  pivot_longer(macroeconomics:religion, names_to = "policy_area") %>% 
  ggplot(aes(value, policy_area, fill = war)) +
  geom_col(position = "dodge")


custom_dict <- dictionary(list(russia = c("russia", "russian", "moscow", "kremlin"),
                               food = c("borscht", "chicken kiev", "shuba"),
                               sport = c("football")))

russia <- tokens_lookup(tokens_ukraine, custom_dict) %>%
  dfm()
russia <- convert(russia, to = "data.frame")
df_wrussia <- cbind(df_ukraine, russia)

df_wrussia %>%
  group_by(war) %>%
  summarise(russia = sum(russia),
            food = sum(food),
            sport = sum(sport))


####################################################################################
## Chinese Text Analysis                                                          ##
####################################################################################

# Doing text analysis can be tricky, but doing Chinese text analysis can be even trickier
# There are a few extra things we need to consider

######################### Showing Chinese Figures #########################
df_ios <-  read.csv("ios.csv")
corpus_ios <- corpus(df_ios, text_field = "message")

# Creating DFM
tokens_ios <- tokens(corpus_ios,
                     remove_punct = TRUE,
                     remove_numbers = TRUE,
                     remove_url = TRUE,
                     remove_symbols = TRUE,
                     verbose = TRUE)
dfm_ios <- dfm(tokens_ios)

customstopwords <- c("與", "年", "月", "日")

dfm_ios <- dfm_remove(dfm_ios, c(stopwords('chinese', source = "misc"), stopwords('english'), customstopwords))

# Inspecting the results again
topfeatures(dfm_ios, 30)

textplot_wordcloud(dfm_ios)

# You would notice Chinese characters are not displayed correctly in the previous graph.
# This is because the default font in R doesn't support Chinese
# To fix it, we simply need to change the font
# Changing font in R can be tricky, but luckily these packages make it easier
library(sysfonts)
font_add_google("Noto Sans TC", "Noto Sans TC")
library(showtext)
showtext_auto()

textplot_wordcloud(dfm_ios)


######################### Tokenisation #########################

# Tokenisation is the process of splitting a string into tokens (in most cases, one token is one word)
# In Latin languages, tokenisation is much easier since splitting the sentence by the space will do the trick
# Since there is no space

text <- "one two 福利 one two 福利 happy everyday one two 福利 one two 福利
福利熊 熊福利 熊福利 福利熊"

# Using the default quanteda tokens() function
print(tokens(text), max_ntoken = 100)

# It works, but not perfectly
# There is another way to do it: Jieba
# Jieba is one of the most widely used packages for tokenisation
# One advantage of using Jieba is that you can add your own words into the dictionary

library(jiebaR)

# Vanila version
tokeniser <- worker()
segment(text, tokeniser)

# Adding user defined words
new_user_word(tokeniser, c("福利熊","熊福利"))
segment(text, tokeniser)

edit_dict()

# Using jieba with quanteda needs a bit of hacking
raw_texts <- as.character(corpus_ios) # getting text from the corpus
tokenised_texts <- purrr::map(raw_texts, segment, tokeniser) # applying jieba::segment() function across rows
tokens_ios <- tokens(tokenised_texts, # Changing it back to a tokens object
                     remove_punct = TRUE,
                     remove_numbers = TRUE,
                     remove_url = TRUE,
                     remove_symbols = TRUE,
                     verbose = TRUE)
tokens_ios

new_user_word(tokeniser, c("今日","台灣"))

dfm_ios <- dfm(tokens_ios)
dfm_ios <- dfm_remove(dfm_ios, c(stopwords('chinese', source = "misc"), stopwords('english'), customstopwords))

topfeatures(dfm_ios, 30)
textplot_wordcloud(dfm_ios)