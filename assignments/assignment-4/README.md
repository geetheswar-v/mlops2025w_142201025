# Assignment 4 - Database Implementation

| Roll Number | Name                  |
|-------------|-----------------------|
| 142201025   | Vedachalam Geetheswar |

# Description

This assignment is based **Databases** and different types such *SQL*, *NOSQL* and Performance in different approches in *NOSQL*.

- Uses **MangoDB** and loading databses in **2 approches** and check performance on 2 approches
- Uses in-memory database **SQLite** for **Relation DBMS** with **2NF** approch
- Uses **Requests** for loading **Online Retail Dataset**

Key features:
- SQLite with 2NF normalization
- MongoDB with transaction-centric and customer-centric approaches
- Performance analysis of different data modeling strategies
- MongoDB Atlas deployment

## Dataset

Online Retail Dataset from UCI Machine Learning Repository:  
[https://archive.ics.uci.edu/dataset/352/](https://archive.ics.uci.edu/dataset/352/)

## Requirements

- Python 3.12+
- uv (Python package manager)
- MongoDB (local installation)
- MongoDB Atlas account (for cloud deployment)

## Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/geetheswar-v/mlops2025w_142201025.git
cd mlops2025w_142201025/assignments/assignment-4
uv sync
```

## Configuration

Create a `.env` file for MongoDB Atlas credentials:

```bash
ATLAS_USERNAME=your_username
ATLAS_PASSWORD=your_password
```

Configuration is managed in `config.toml`.

## Usage

### 1. Prepare Data (Prereqsites for all questions)

Generate a dataset:

```bash
uv run prepare_data
```

### 2. SQLite Database (2NF)

Create and populate SQLite database:

```bash
# Create schema
uv run prepare_db

# Populate with data
uv run populate_db
```

### 3. MongoDB - Local Deployment

Ingest data into local MongoDB (both approaches):

```bash
uv run ingest_collections --local
```

This creates:
- `transactions` collection (transaction-centric)
- `customers` collection (customer reference data)
- `customers_centric` collection (customer-centric)

### 4. MongoDB - Atlas Deployment

Ingest data into MongoDB Atlas (transaction-centric only):

```bash
uv run ingest_collections --atlas
```

### 5. Run Performance Tests

Execute CRUD operations and measure performance:

```bash
uv run performance_test
```

## Assignment Questions

1. **Question 1:** SQLite database with 2NF normalization
2. **Question 2:** MongoDB document-oriented approaches (transaction-centric and customer-centric)
3. **Question 3:** CRUD operations performance analysis (see `Assignment.md`)
4. **Question 4:** MongoDB Atlas deployment

## Acknowledgment

Dataset provided by UCI Machine Learning Repository.

**Citation:**
```bibtex
@misc{online_retail_352,
  author       = {Chen, Daqing},
  title        = {{Online Retail}},
  year         = {2015},
  howpublished = {UCI Machine Learning Repository},
  note         = {{DOI}: https://doi.org/10.24432/C5BW33}
}
```
