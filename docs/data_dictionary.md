# Data dictionary

Owner: Member 1. Fill in from the Kaggle **Data** tab and your own checks.
Columns marked *(verify)* were not confirmed from the official page.

## transactions_train.csv (~31M rows)
| Column | Meaning | Notes |
|---|---|---|
| `t_dat` | purchase date | |
| `customer_id` | customer key | joins to customers.csv |
| `article_id` | product key | joins to articles.csv |
| `price` | price paid | appears scaled, not real currency *(verify)* |
| `sales_channel_id` | online or store | *(verify which number is which)* |

## customers.csv
| Column | Meaning | Notes |
|---|---|---|
| `customer_id` | customer key | |
| `age` | age in years | check missing values |
| fashion-news status | subscribed to Fashion News | *(verify column name)* |
| ... | | |

## articles.csv (~105k rows, ~25 columns)
| Column | Meaning | Notes |
|---|---|---|
| `article_id` | product key | |
| product type / group | | |
| colour, pattern (graphical appearance) | | |
| department / index | | |
| `detail_desc` | free-text description | used by Member 4 *(verify)* |
