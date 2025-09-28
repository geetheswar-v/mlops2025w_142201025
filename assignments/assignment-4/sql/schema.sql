-- SQLite script to create tables for the Online Retail dataset in 2NF.
-- This schema separates products, invoices, and transaction items.
-- Author: Geetheswar
-- Date: 2025-10-28

-- Splitting Idea
-- (InvoiceNo, StockCode) is the composite primary key in 1NF
-- Description is dependent on StockCode
-- InvoiceDate, CustomerID is dependent on InvoiceNo
-- partial dependencies exist in this table
-- so we can split the table into 3 tables
-- Products(StockCode, Description)
-- Invoices(InvoiceNo, InvoiceDate, CustomerID, Country)
-- InvoiceItems(InvoiceNo, StockCode, Quantity, UnitPrice)

-- Enable foreign key constraint enforcement in SQLite.
PRAGMA foreign_keys = ON;

-- Table for product-specific information
CREATE TABLE Products (
    StockCode   INTEGER PRIMARY KEY,
    Description TEXT
);

-- Table for invoice header information
CREATE TABLE Invoices (
    InvoiceNo   TEXT PRIMARY KEY,
    InvoiceDate DATETIME NOT NULL,
    CustomerID  INTEGER,
    Country     TEXT
);

-- Linking table for each line item of an invoice
CREATE TABLE InvoiceItems (
    InvoiceNo TEXT NOT NULL,
    StockCode INTEGER NOT NULL,
    Quantity  INTEGER NOT NULL,
    UnitPrice REAL NOT NULL,
    PRIMARY KEY (InvoiceNo, StockCode),
    FOREIGN KEY (InvoiceNo) REFERENCES Invoices(InvoiceNo),
    FOREIGN KEY (StockCode) REFERENCES Products(StockCode)
);

-- I created this SQL such a way that it violates 3NF but adheres to 2NF.
-- This database violates 3NF in Invoices Table.
-- as Country is dependent on CustomerID
-- where CustomerID is dependent on InvoiceNo (Primary Key)
-- Which is a transitive dependency.

-- InvoiceNo -> CustomerID -> Country (Transitive Dependency)