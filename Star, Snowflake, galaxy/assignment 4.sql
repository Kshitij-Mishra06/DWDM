create database dwdm_assig4;

use dwdm_assig4;


-- This is for star schema.

create table Fact_Transction(
    Transaction_ID int not null,
    Account_ID int not null,
    Branch_ID int not null,
    Date_ID int not null,
    Product_ID int not null,
    primary key (Transaction_ID),
    foreign key (Account_ID) references Dim_Account(Account_ID),
    foreign key (Branch_ID) references Dim_Branch(Branch_ID),
    foreign key (Date_ID) references Dim_Time(Date_ID),
    foreign key (Product_ID) references Dim_Product(Product_ID)
);

create table Dim_Account(
    Account_ID int not null,
    Account_Type varchar(50),
    Account_Status varchar(50),
    primary key(Account_ID)
);

create table Dim_Branch (
    Branch_ID int not null,
    Branch_Name varchar(100),
    Location varchar(255),
    primary key (Branch_ID)
);

create table Dim_Time (
    Date_ID int not null,
    Date int,
    Month int,
    Quarter int,
    Year int,
    primary key (Date_ID)
);

create table Dim_Product (
    Product_ID int not null,
    Product_Type varchar(50),
    primary key(Product_ID)
);

-- This is for snowflake schema.

CREATE TABLE Dim_Branch_Location (
    Location_ID int NOT NULL,
    City varchar(100),
    State varchar(100),
    Country varchar(100),
    primary key (Location_ID)
);


ALTER TABLE Dim_Branch
ADD Location_ID int,
ADD CONSTRAINT fk_location
FOREIGN KEY (Location_ID) REFERENCES Dim_Branch_Location(Location_ID);


-- This is for galaxy schema. 

CREATE TABLE Fact_Customer_Transaction (
    Customer_Transaction_ID int NOT NULL,
    Customer_ID int NOT NULL,
    Branch_ID int NOT NULL,
    Date_ID int NOT NULL,
    Product_ID int NOT NULL,
    Amount decimal(18, 2),
    primary key (Customer_Transaction_ID),
    foreign key (Branch_ID) references Dim_Branch(Branch_ID),
    foreign key (Date_ID) references Dim_Time(Date_ID),
    foreign key (Product_ID) references Dim_Product(Product_ID)
);

CREATE TABLE Dim_Customer (
    Customer_ID int NOT NULL,
    Customer_Name varchar(100),
    Customer_Age int,
    Customer_Gender varchar(10),
    primary key (Customer_ID)
);


ALTER TABLE Fact_Customer_Transaction
ADD CONSTRAINT fk_customer
FOREIGN KEY (Customer_ID) REFERENCES Dim_Customer(Customer_ID);

-- entering data.

-- Inserting data for Dim_Account (Star Schema)
INSERT INTO Dim_Account (Account_ID, Account_Type, Account_Status) VALUES
(1, 'Savings', 'Active'),
(2, 'Checking', 'Inactive'),
(3, 'Loan', 'Active'),
(4, 'Savings', 'Closed'),
(5, 'Business', 'Active');

-- Inserting data for Dim_Branch_Location (Snowflake Schema)
INSERT INTO Dim_Branch_Location (Location_ID, City, State, Country) VALUES
(1, 'New York', 'NY', 'USA'),
(2, 'Los Angeles', 'CA', 'USA'),
(3, 'Chicago', 'IL', 'USA'),
(4, 'Miami', 'FL', 'USA'),
(5, 'San Francisco', 'CA', 'USA');

-- Inserting data for Dim_Branch (Star Schema)
INSERT INTO Dim_Branch (Branch_ID, Branch_Name, Location, Location_ID) VALUES
(1, 'Main Branch', 'New York', 1),
(2, 'West Branch', 'Los Angeles', 2),
(3, 'East Branch', 'Chicago', 3),
(4, 'South Branch', 'Miami', 4),
(5, 'North Branch', 'San Francisco', 5);

-- Inserting data for Dim_Time (Star Schema)
INSERT INTO Dim_Time (Date_ID, Date, Month, Quarter, Year) VALUES
(1, 20240115, 1, 1, 2024),
(2, 20240220, 2, 1, 2024),
(3, 20240530, 5, 2, 2024),
(4, 20240715, 7, 3, 2024),
(5, 20240801, 8, 3, 2024),
(6, 20240913, 9, 3, 2024);

-- Inserting data for Dim_Product (Star Schema)
INSERT INTO Dim_Product (Product_ID, Product_Type) VALUES
(1, 'Loan'),
(2, 'Credit Card'),
(3, 'Savings Account'),
(4, 'Investment Account'),
(5, 'Fixed Deposit');

-- Inserting data for Dim_Customer (Galaxy Schema)
INSERT INTO Dim_Customer (Customer_ID, Customer_Name, Customer_Age, Customer_Gender) VALUES
(1, 'John Doe', 30, 'Male'),
(2, 'Jane Smith', 40, 'Female'),
(3, 'Alex Johnson', 25, 'Non-binary'),
(4, 'Emily Davis', 35, 'Female'),
(5, 'Michael Brown', 28, 'Male');

-- Inserting data for Fact tables (after Dimension tables are populated)

-- Inserting data for Fact_Transction (Star Schema)
INSERT INTO Fact_Transction (Transaction_ID, Account_ID, Branch_ID, Date_ID, Product_ID) VALUES
(1, 1, 1, 1, 1),
(2, 2, 2, 2, 2),
(3, 3, 3, 3, 3),
(4, 4, 4, 4, 4),
(5, 5, 5, 5, 5);

-- Inserting data for Fact_Customer_Transaction (Galaxy Schema)
INSERT INTO Fact_Customer_Transaction (Customer_Transaction_ID, Customer_ID, Branch_ID, Date_ID, Product_ID, Amount) VALUES
(1, 1, 1, 1, 1, 10000.00),
(2, 2, 2, 2, 2, 250.50),
(3, 3, 3, 3, 3, 1200.75),
(4, 4, 4, 4, 4, 5000.00),
(5, 5, 5, 5, 5, 750.25);

-- Queries.

-- star schema query

SELECT 
    f.Transaction_ID, 
    a.Account_Type, 
    b.Branch_Name, 
    p.Product_Type, 
    t.Date, t.Month, t.Year
FROM 
    Fact_Transction f
JOIN 
    Dim_Account a ON f.Account_ID = a.Account_ID
JOIN 
    Dim_Branch b ON f.Branch_ID = b.Branch_ID
JOIN 
    Dim_Product p ON f.Product_ID = p.Product_ID
JOIN 
    Dim_Time t ON f.Date_ID = t.Date_ID;

-- snowflake schema

SELECT 
    b.Branch_ID, 
    b.Branch_Name, 
    l.City, 
    l.State, 
    l.Country
FROM 
    Dim_Branch b
JOIN 
    Dim_Branch_Location l ON b.Location_ID = l.Location_ID;


-- galaxy schema.

SELECT 
    f.Customer_Transaction_ID, 
    c.Customer_Name, 
    c.Customer_Age, 
    b.Branch_Name, 
    p.Product_Type, 
    f.Amount, 
    t.Date, t.Month, t.Year
FROM 
    Fact_Customer_Transaction f
JOIN 
    Dim_Customer c ON f.Customer_ID = c.Customer_ID
JOIN 
    Dim_Branch b ON f.Branch_ID = b.Branch_ID
JOIN 
    Dim_Product p ON f.Product_ID = p.Product_ID
JOIN 
    Dim_Time t ON f.Date_ID = t.Date_ID;
