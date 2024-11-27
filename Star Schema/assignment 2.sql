create database Star_Schema;

use Star_Schema;

-- creating Fact_sales table.

create table Fact_Sales ( 
Sales_Rep_id INT NOT NULL, 
Time_id INT NOT NULL,
Order_id INT NOT NULL, 
Customer_id INT NOT NULL, 
Product_id INT NOT NULL,
Net_amount_per_customer INT NOT NULL, 
Net_amount_per_product INT NOT NULL, 
Net_amount_per_promotion INT NOT NULL
);

-- Inserting values into fact sales table.

INSERT INTO Fact_Sales VALUES
('1', '7','2', '3', '6', '0', '0','0'),
('2', '9','10', '5', '8','0', '0','0'),
('3', '6','5', '7', '1', '0', '0','0'),
('4', '1','8', '9', '3', '0', '0','0'),
('5', '10','7', '1', '9', '0', '0','0'),
('6', '4','1', '10', '7','0', '0','0'),
('7', '8','3', '8', '10','0', '0','0'),
('8', '2','4', '6', '2', '0', '0','0'),
('9', '5','6', '4', '5', '0', '0','0'),
('10', '3','9', '2', '4', '0', '0','0');

select * from Fact_Sales;


-- creating Dim sales rep table.

create table Dim_Sales_Rep( 
Sales_Rep_id INT NOT NULL,
Name varchar(30) NOT NULL,
Deal varchar(30) NOT NULL, 
Discount INT NOT NULL, 
primary key(Sales_Rep_id)
);

-- inserting values in dim sales rep.

INSERT INTO Dim_Sales_Rep VALUES
('1', 'Kshitij Mishra','Yes', '20'),
('2', 'Abdul_Bari Mulla','Yes', '15'),
('3', 'Om Gaadiya','No', '30'),
('4', 'Chinmay Maheshwari','No', '7'),
('5', 'Shubham Patel','Yes', '10'),
('6', 'Lobhas Pansare','Yes', '13'),
('7', 'Sachin Tendulkar','No', '19'),
('8', 'Amit Shah','Yes', '10'),
('9', 'Ruturaj Gaikwad','No', '8'),
('10', 'Rohit Mahadik','Yes', '15')

drop table Dim_Sales_Rep;

select * from Dim_Sales_Rep;

create table Dim_Time ( 
Time_id int not null,
day varchar(30) not null, 
month varchar(30) not null, 
year int not null,
primary key(Time_id)
);

INSERT INTO Dim_Time VALUES
('1', 'Monday','Jan', '2010'),
('2', 'Tuesday','Feb', '2011'),
('3', 'Wednesday','March', '2012'),
('4', 'Thrusday','April', '2016'),
('5', 'Friday','May', '2017'),
('6', 'Monday','June', '2019'),
('7', 'Tuesday','July', '2011'),
('8', 'Wednesday','Oct', '2014'),
('9', 'Thrusday','Nov', '2016'),
('10', 'Friday','Dec', '2021');

select * from Dim_Time;

create table Dim_Customer ( 
Customer_id int not null,
Name varchar(30) not null, 
billing_address varchar(60) not null, 
shipping_address varchar(60) not null, 
primary key(Customer_id)
);

INSERT INTO Dim_Customer VALUES
('1', 'Neha','abc', 'abc'),
('2', 'Ved','def', 'def'),
('3', 'Sameer','ghi', 'ghi'),
('4', 'Taniya','jki', 'jki'),
('5', 'Anvi','mno', 'mno'),
('6', 'Rohit','pqr', 'pqr'),
('7', 'Ayansh','stu', 'xyz'),
('8', 'Anjali','vu', 'stu'),
('9', 'Piyu','abc', 'xyz'),
('10', 'Anju','xyz', 'xyz');

select * from Dim_Customer;


create table Dim_Product ( 
Product_id int not null,
product_number int not null, 
type varchar(30) not null, 
quality varchar(30) not null, 
price int not null,
requested_ship_date date not null, 
primary key(Product_id)
);

INSERT INTO Dim_Product VALUES
('1','101','Mouse','A1','700','2022-09-16'),
('2','102','Keyboard','A1','1500','2022-08-09'),
('3','103','RAM','A1','5000','2022-08-01'),
('4','104','USB cable','A1','600','2022-07-23'),
('5','105','Pendrive','A1','800','2022-07-11'),
('6','106','Hard Disk','A1','4000','2022-06-24'),
('7','107','Charger','A1','500','2022-04-08'),
('8','108','Ethernet Cable','A1','1200','2022-03-12'),
('9','109','Graphic Card','A1','6000','2022-02-16'),
('10','110','Printer','A1','6500','2022-01-01');

select * from Dim_Product;

create table Dim_Order ( 
Order_id int not null,
order_number int not null, 
order_date date not null, 
amount int not null, 
primary key(Order_id)
);

INSERT INTO Dim_Order VALUES
('1','1001','2022-09-16','1'),
('2','1002','2022-08-09','2'),
('3','1003','2022-08-01','6'),
('4','1004','2022-07-23','1'),
('5','1005','2022-07-11','2'),
('6','1006','2022-06-24','3'),
('7','1007','2022-04-08','1'),
('8','1008','2022-03-12','3'),
('9','1009','2022-02-16','4'),
('10','1010','2022-01-01','1');

select * from Dim_Order;

-- Adding primary keys to the dimension tables
ALTER TABLE Dim_Sales_Rep
ADD PRIMARY KEY (Sales_Rep_id);

ALTER TABLE Dim_Time
ADD PRIMARY KEY (Time_id);

ALTER TABLE Dim_Customer
ADD PRIMARY KEY (Customer_id);

ALTER TABLE Dim_Product
ADD PRIMARY KEY (Product_id);

ALTER TABLE Dim_Order
ADD PRIMARY KEY (Order_id);

-- Adding foreign keys to the Fact_Sales table
ALTER TABLE Fact_Sales
ADD CONSTRAINT fk_sales_rep
FOREIGN KEY (Sales_Rep_id) REFERENCES Dim_Sales_Rep(Sales_Rep_id);

ALTER TABLE Fact_Sales
ADD CONSTRAINT fk_time
FOREIGN KEY (Time_id) REFERENCES Dim_Time(Time_id);

ALTER TABLE Fact_Sales
ADD CONSTRAINT fk_order
FOREIGN KEY (Order_id) REFERENCES Dim_Order(Order_id);

ALTER TABLE Fact_Sales
ADD CONSTRAINT fk_customer
FOREIGN KEY (Customer_id) REFERENCES Dim_Customer(Customer_id);

ALTER TABLE Fact_Sales
ADD CONSTRAINT fk_product
FOREIGN KEY (Product_id) REFERENCES Dim_Product(Product_id);


-- queries 

-- Net Order Dollar Amount per Customer:

SELECT 
    c.Customer_id,
    c.Name AS Customer_Name,
    SUM(f.Net_amount_per_customer) AS Total_Net_Amount_Per_Customer
FROM 
    Fact_Sales f
JOIN 
    Dim_Customer c ON f.Customer_id = c.Customer_id
GROUP BY 
    c.Customer_id, c.Name;

-- net Order Dollar Amount per Product:

SELECT 
    p.Product_id,
    p.type AS Product_Type,
    SUM(f.Net_amount_per_product) AS Total_Net_Amount_Per_Product
FROM 
    Fact_Sales f
JOIN 
    Dim_Product p ON f.Product_id = p.Product_id
GROUP BY 
    p.Product_id, p.type;


-- Net Order Dollar Amount per Promotion:

SELECT 
    f.Sales_Rep_id,
    s.Name AS Sales_Rep_Name,
    SUM(f.Net_amount_per_promotion) AS Total_Net_Amount_Per_Promotion
FROM 
    Fact_Sales f
JOIN 
    Dim_Sales_Rep s ON f.Sales_Rep_id = s.Sales_Rep_id
GROUP BY 
    f.Sales_Rep_id, s.Name;



-- Net Order Dollar Amount and Ad Performance of Sales Representatives:

SELECT 
    s.Sales_Rep_id,
    s.Name AS Sales_Rep_Name,
    SUM(f.Net_amount_per_customer + f.Net_amount_per_product + f.Net_amount_per_promotion) AS Total_Net_Amount,
    s.Deal,
    s.Discount
FROM 
    Fact_Sales f
JOIN 
    Dim_Sales_Rep s ON f.Sales_Rep_id = s.Sales_Rep_id
GROUP BY 
    s.Sales_Rep_id, s.Name, s.Deal, s.Discount;





