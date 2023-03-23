# Load packages ===============================================================
library(tidyverse)
library(gutenbergr) # Books!
library(tm) # Text mining objects
library(textstem) # Stemming and lemmatizing utilities
library(topicmodels)
library(tidytext) # Tidy wrangling and cleaning tools for text data
library(tidymodels)

# Data Acquisition and Preprocessing ==========================================
# Download data
titles <- c("Twenty Thousand Leagues under the Sea",
            "Pride and Prejudice",
            "Great Expectations")
books <- gutenbergr::gutenberg_works(title %in% titles) %>%
  gutenbergr::gutenberg_download(meta_fields = "title")

# Split into chapters and generate document-term matrix ------------------------
chapters <- books %>%
  dplyr::group_by(title) %>%
  dplyr::mutate(
    chapter = cumsum(
      stringr::str_detect(
        text, stringr::regex("^chapter ", ignore_case = TRUE)
      )
    )
  ) %>%
  dplyr::ungroup() %>%
  dplyr::filter(chapter > 0) %>%
  tidyr::unite(document, title, chapter)
(chapter_words <- tidytext::unnest_tokens(chapters, word, text))

word_counts <- chapter_words %>%
  dplyr::anti_join(tidytext::stop_words) %>%
  dplyr::count(document, word, sort = TRUE)
(dtm <- tidytext::cast_dtm(word_counts, document, word, n))

# Topic modeling ==============================================================
(chapters_lda <- topicmodels::LDA(dtm, k = 3, control = list(seed = 1234)))
topics <- tidytext::tidy(chapters_lda, matrix = "beta")
topics %>%
  dplyr::group_by(topic) %>%
  dplyr::slice_max(beta, n = 5) %>%
  dplyr::ungroup() %>%
  dplyr::mutate(term = tidytext::reorder_within(term, beta, topic)) %>%
  ggplot2::ggplot(aes(x = beta, y = term, fill = factor(topic))) +
  ggplot2::geom_col(show.legend = FALSE) +
  ggplot2::facet_wrap(~ topic, scales = "free") +
  tidytext::scale_y_reordered()
mixtures <- chapters_lda %>%
  tidytext::tidy(matrix = "gamma") %>%
  dplyr::arrange(document)

# Classification ==============================================================
# Train-test split ------------------------------------------------------------
split <- mixtures %>%
  dplyr::mutate(
    topic = paste0("Topic_", topic),
    book = stringr::str_extract(document, ".+(?=_)")
  ) %>%
  tidyr::pivot_wider(names_from = "topic", values_from = "gamma") %>%
  rsample::initial_split(prop = 0.5, strata = book)
train <- rsample::training(split)
test <- rsample::testing(split)

# Train knn classifier --------------------------------------------------------\
knn_model <- parsnip::nearest_neighbor(dist_power = 1) %>%
  parsnip::set_engine("kknn") %>%
  parsnip::set_mode("classification")
knn_recipe <- recipes::recipe(
  book ~ Topic_1 + Topic_2 + Topic_3, data = train
)
knn_wflow <- workflows::workflow() %>%
  workflows::add_model(knn_model) %>%
  workflows::add_recipe(knn_recipe)
knn_fit <- generics::fit(knn_wflow, data = train)

# Evaluate results ------------------------------------------------------------
(predictions <- dplyr::bind_cols(test, predict(knn_fit, test)))
predictions %>%
  dplyr::filter(book == .pred_class) %>%
  nrow() %>%
  `/`(nrow(test))