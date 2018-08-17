library(tidyverse)
library(rvest)
library(stringr)
library(magrittr)

extractSeason <- function(link) {
  if (startsWith(link, "10")) {
    10
  } else {
    str_split(link, "season|/")[[1]][2] %>% as.numeric()
  }
}

extractTitle <- function(season, html) {
  title <- html_nodes(html, "title") %>% html_text() %>% paste(collapse = " ")
  if (season == 10) {
    title <- str_split(title, " - ")[[1]][3]
  }
  if (season != 9 & length(title) > 0) {
    title
  } else {
    ""
  }
}

getSeason9Titles <- function() {
  titles <- read_html("https://en.wikipedia.org/wiki/Friends_(season_9)") %>%
    html_nodes(".summary") %>%
    html_text()
  map_chr(titles[4:26], function(x) str_split(x, "\"")[[1]][2])
}

url <- "http://livesinabox.com/friends/scripts.shtml"

episodes_df <- read_html(url) %>%
  html_nodes("a") %>%
  html_attr("href") %>%
  tibble(link = .) %>%
  slice(46:275) %>%
  unique() %>%
  mutate(season = map_dbl(link, extractSeason),
         html = map(paste0("http://livesinabox.com/friends/", link), read_html),
         episodeTitle = map2_chr(season, html, extractTitle)) %>%
  filter(!startsWith(episodeTitle, "Friends")) %>%
  group_by(season) %>%
  mutate(episodeNum = row_number()) %>%
  ungroup()

episodes_df$episodeTitle[episodes_df$season == 9] <- getSeason9Titles()

episodes_df %>% select(-link)

getPeronLinePairs <- function(html) {
  html %>%
    html_nodes("body") %>%
    html_nodes("p") %>%
    html_text() %>%
    tibble(text = .) %>%
    filter(str_detect(text, "^[A-Z][a-zA-Z. ]+:")) %>%
    unlist() %>%
    unname() %>%
    str_to_lower() %>%
    str_replace_all("\n", " ") %>%
    str_replace(":", "\\|\\|")
}

getPeronLinePairsSeasonIrregulars <- function(html) {
  html %>%
    html_nodes("body") %>%
    html_text() %>%
    str_split(., "\n") %>%
    unlist %>%
    tibble(text = .) %>%
    filter(str_detect(text, "^[A-Z][a-zA-Z. ]+:")) %>%
    unlist() %>%
    unname() %>%
    str_to_lower() %>%
    str_replace_all("\n", " ") %>%
    str_replace(":", "\\|\\|")
}

personLines_df <- episodes_df %>%
  filter(!(season == 2 & episodeNum %in% c(9, 12:23)) &
           !(season == 9 & episodeNum %in% c(7, 11, 15))) %>%
  mutate(personLine = map(html, getPeronLinePairs))

irregulars <- episodes_df %>%
  filter((season == 2 & episodeNum %in% c(9, 12:23)) |
           (season == 9 & episodeNum %in% c(7, 11, 15))) %>%
  mutate(personLine = map(html, getPeronLinePairsSeasonIrregulars))

personLines_df %<>%
  rbind(irregulars) %>%
  group_by(season, episodeNum, episodeTitle) %>%
  unnest(personLine) %>%
  ungroup() %>%
  separate(personLine, c("person", "line"), sep = "\\|\\|") %>%
  filter(!str_detect(person, " by"))

personLines_df %>% select(season, episodeNum, person, line)

write.table(personLines_df, file = "dialogue.csv", sep="|")
