# Data dictionary

Owner: Member 1 (Data & Evaluation Lead).
This document describes the H&M datasets used in our recommendation system, including their variables and data preparation.

## transactions_train.csv (~31M rows)
| Column | Meaning | Notes |
|---|---|---|
| `t_dat` | purchase date | |
| `customer_id` | customer key | joins to customers.csv |
| `article_id` | product key | joins to articles.csv |
| `price` | Purchase price | Values appear scaled; exact units are not confirmed |
| `sales_channel_id` | Sales channel | 1 = store, 2 = online (commonly reported, not officially confirmed) |

## customers.csv (~1.37M rows)
| Column | Meaning | Notes |
|---|---|---|
| `customer_id` | Customer key | Unique customer identifier |
| `FN` | Customer metadata flag | Exact meaning not confirmed |
| `Active` | Customer metadata flag | Exact meaning not confirmed |
| `club_member_status` | Membership status | Categorical variable |
| `fashion_news_frequency` | Fashion news frequency | 16,011 missing values |
| `age` | Customer age in years | 15,861 missing values |
| `postal_code` | Anonymized postal code | Customer location identifier |

## articles.csv (~105k rows, 25 columns)
| Column | Meaning | Notes |
|---|---|---|
| `article_id` | Product key | Joins to transactions_train.csv |
| `product_code`, `prod_name` | Product code and name | Product identification |
| `product_type_no`, `product_type_name` | Product type | Product classification |
| `product_group_name` | Product group | Used for diversity evaluation |
| `graphical_appearance_no`, `graphical_appearance_name` | Product pattern | Visual attribute |
| `colour_group_code`, `colour_group_name` | Colour group | Product colour |
| `perceived_colour_value_id`, `perceived_colour_value_name` | Perceived colour value | Colour attribute |
| `perceived_colour_master_id`, `perceived_colour_master_name` | Main perceived colour | Colour classification |
| `department_no`, `department_name` | Department | Product classification |
| `index_code`, `index_name` | Product index | Product classification |
| `index_group_no`, `index_group_name` | Index group | Product classification |
| `section_no`, `section_name` | Section | Product classification |
| `garment_group_no`, `garment_group_name` | Garment group | Clothing category |
| `detail_desc` | Free-text product description | Useful for content-based recommendations |

## Data preparation

We use the most recent 12 weeks of transactions for training and the final 7 days for testing.

| Dataset | Period | Transactions |
|---|---|---|
| Training | 2020-06-24 to 2020-09-15 | 3,664,080 |
| Testing | 2020-09-16 to 2020-09-22 | 240,311 |

The datasets were split chronologically to prevent future transactions from being included in training. The processed files are saved in Parquet format.

## Customer segments
Customers are classified based on their number of purchases during training.
| Segment | Definition |
|---|---|
| New | 0 purchases during the 12-week training period |
| Light | 1–4 training purchases |
| Regular | 5 or more training purchases |

## New-item cold start
The testing period contains 922 products that were not purchased during training.
These products account for 9,904 transactions, representing 4.12% of all testing transactions.

## Data quality and leakage checks
No missing values were found in the five transaction columns.
The training period contains 84 days and the testing period contains 7 days, with no overlap between the dates.
Customer segments and item popularity are calculated from training data only. Additional leakage checks are needed when evaluating individual recommendation models.
The original datasets are excluded from GitHub because of their size and access requirements.