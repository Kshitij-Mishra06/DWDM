create database DWDM_assig5;

use DWDM_assig5;


CREATE TABLE Dim_Patient (
    Patient_ID INT PRIMARY KEY AUTO_INCREMENT,
    First_Name VARCHAR(50),
    Last_Name VARCHAR(50),
    DOB DATE,
    Gender VARCHAR(10),
    Address VARCHAR(255),
    Contact_Info VARCHAR(50),
    Insurance_Provider VARCHAR(100),
    Policy_Number VARCHAR(50)
);

CREATE TABLE Dim_Doctor (
    Doctor_ID INT PRIMARY KEY AUTO_INCREMENT,
    First_Name VARCHAR(50),
    Last_Name VARCHAR(50),
    Specialization VARCHAR(100),
    Department VARCHAR(100),
    Contact_Info VARCHAR(50)
);

CREATE TABLE Dim_Service (
    Service_ID INT PRIMARY KEY AUTO_INCREMENT,
    Service_Type VARCHAR(50),
    Room_Number INT,
    Bed_Number INT
);



CREATE TABLE Dim_Medication (
    Medication_ID INT PRIMARY KEY AUTO_INCREMENT,
    Drug_Name VARCHAR(100),
    Dosage VARCHAR(50),
    Frequency VARCHAR(50),
    Start_Date DATE,
    End_Date DATE
);

CREATE TABLE Dim_Date (
    Date_Key INT PRIMARY KEY AUTO_INCREMENT,
    Full_Date DATE,
    Day INT,
    Month INT,
    Year INT
);

CREATE TABLE Dim_Date (
    Date_Key INT PRIMARY KEY AUTO_INCREMENT,
    Full_Date DATE,
    Day INT,
    Month INT,
    Year INT
);

CREATE TABLE Fact_Treatment (
    Treatment_ID INT PRIMARY KEY AUTO_INCREMENT,
    Patient_ID INT,
    Doctor_ID INT,
    Service_ID INT,
    Diagnosis_ID INT,
    Medication_ID INT,
    Date_Key INT,
    Cost DECIMAL(10, 2),
    Insurance_Claim DECIMAL(10, 2),
    Amount_Paid DECIMAL(10, 2),
    FOREIGN KEY (Patient_ID) REFERENCES Dim_Patient(Patient_ID),
    FOREIGN KEY (Doctor_ID) REFERENCES Dim_Doctor(Doctor_ID),
    FOREIGN KEY (Service_ID) REFERENCES Dim_Service(Service_ID),
    FOREIGN KEY (Medication_ID) REFERENCES Dim_Medication(Medication_ID),
    FOREIGN KEY (Date_Key) REFERENCES Dim_Date(Date_Key)
);

CREATE TABLE Dim_Specialization (
    Specialization_ID INT PRIMARY KEY AUTO_INCREMENT,
    Specialization_Name VARCHAR(100)
);

ALTER TABLE Dim_Doctor 
ADD COLUMN Specialization_ID INT,
ADD FOREIGN KEY (Specialization_ID) REFERENCES Dim_Specialization(Specialization_ID);


CREATE TABLE Fact_Appointment (
    Appointment_ID INT PRIMARY KEY AUTO_INCREMENT,
    Patient_ID INT,
    Doctor_ID INT,
    Date_Key INT,
    Appointment_Status VARCHAR(50),
    Appointment_Cost DECIMAL(10, 2),
    FOREIGN KEY (Patient_ID) REFERENCES Dim_Patient(Patient_ID),
    FOREIGN KEY (Doctor_ID) REFERENCES Dim_Doctor(Doctor_ID),
    FOREIGN KEY (Date_Key) REFERENCES Dim_Date(Date_Key)
);



-- Connect Fact_Appointment to Dim_Patient
ALTER TABLE Fact_Appointment 
ADD CONSTRAINT fk_patient FOREIGN KEY (Patient_ID) 
REFERENCES Dim_Patient(Patient_ID);

-- Connect Fact_Appointment to Dim_Doctor
ALTER TABLE Fact_Appointment 
ADD CONSTRAINT fk_doctor FOREIGN KEY (Doctor_ID) 
REFERENCES Dim_Doctor(Doctor_ID);

-- Connect Fact_Appointment to Dim_Date
ALTER TABLE Fact_Appointment 
ADD CONSTRAINT fk_date FOREIGN KEY (Date_Key) 
REFERENCES Dim_Date(Date_Key);


-- Data for the database

-- Insert data into Dim_Patient
INSERT INTO Dim_Patient (First_Name, Last_Name, DOB, Gender, Address, Contact_Info, Insurance_Provider, Policy_Number)
VALUES
('John', 'Doe', '1985-06-15', 'Male', '123 Maple St', '555-1234', 'HealthPlus', 'HP12345'),
('Jane', 'Smith', '1990-08-22', 'Female', '456 Oak St', '555-5678', 'CareHealth', 'CH67890'),
('Bob', 'Johnson', '1975-11-10', 'Male', '789 Pine St', '555-9876', 'MedCare', 'MC54321'),
('Alice', 'Williams', '1982-03-05', 'Female', '321 Birch St', '555-3456', 'InsureHealth', 'IH34567'),
('Tom', 'Brown', '1995-12-12', 'Male', '654 Cedar St', '555-7654', 'CarePlus', 'CP76543');

-- Insert data into Dim_Doctor
INSERT INTO Dim_Doctor (First_Name, Last_Name, Specialization, Department, Contact_Info)
VALUES
('Emily', 'Taylor', 'Cardiology', 'Cardiology', '555-1122'),
('Michael', 'Davis', 'Neurology', 'Neurology', '555-3344'),
('Sarah', 'Miller', 'Orthopedics', 'Orthopedics', '555-5566'),
('James', 'Wilson', 'Pediatrics', 'Pediatrics', '555-7788'),
('Linda', 'Moore', 'Dermatology', 'Dermatology', '555-9900');

-- Insert data into Dim_Service
INSERT INTO Dim_Service (Service_Type, Room_Number, Bed_Number)
VALUES
('Surgery', 101, 1),
('ICU', 102, 2),
('Consultation', 103, 3),
('Radiology', 104, 4),
('Therapy', 105, 5);

-- Insert data into Dim_Medication
INSERT INTO Dim_Medication (Drug_Name, Dosage, Frequency, Start_Date, End_Date)
VALUES
('Aspirin', '500mg', 'Once a day', '2024-09-01', '2024-09-10'),
('Metformin', '850mg', 'Twice a day', '2024-09-05', '2024-09-15'),
('Ibuprofen', '200mg', 'Three times a day', '2024-09-07', '2024-09-12'),
('Lisinopril', '10mg', 'Once a day', '2024-09-10', '2024-09-20'),
('Atorvastatin', '20mg', 'Once a day', '2024-09-12', '2024-09-22');

-- Insert data into Dim_Date
INSERT INTO Dim_Date (Full_Date, Day, Month, Year)
VALUES
('2024-09-01', 1, 9, 2024),
('2024-09-05', 5, 9, 2024),
('2024-09-07', 7, 9, 2024),
('2024-09-10', 10, 9, 2024),
('2024-09-12', 12, 9, 2024);

-- Insert data into Dim_Specialization
INSERT INTO Dim_Specialization (Specialization_Name)
VALUES
('Cardiology'),
('Neurology'),
('Orthopedics'),
('Pediatrics'),
('Dermatology');

-- Insert data into Fact_Treatment
INSERT INTO Fact_Treatment (Patient_ID, Doctor_ID, Service_ID, Medication_ID, Date_Key, Cost, Insurance_Claim, Amount_Paid)
VALUES
(1, 1, 1, 1, 1, 200.00, 150.00, 50.00),
(2, 2, 2, 2, 2, 500.00, 400.00, 100.00),
(3, 3, 3, 3, 3, 150.00, 100.00, 50.00),
(4, 4, 4, 4, 4, 300.00, 250.00, 50.00),
(5, 5, 5, 5, 5, 400.00, 300.00, 100.00);

-- Insert data into Fact_Appointment
INSERT INTO Fact_Appointment (Patient_ID, Doctor_ID, Date_Key, Appointment_Status, Appointment_Cost)
VALUES
(1, 1, 1, 'Completed', 150.00),
(2, 2, 2, 'Cancelled', 0.00),
(3, 3, 3, 'Completed', 100.00),
(4, 4, 4, 'Scheduled', 200.00),
(5, 5, 5, 'Completed', 250.00);


-- Updating Dim_Doctor to add the correct Specialization_ID
UPDATE Dim_Doctor
SET Specialization_ID = (SELECT Specialization_ID FROM Dim_Specialization WHERE Specialization_Name = 'Cardiology')
WHERE First_Name = 'Emily' AND Last_Name = 'Taylor';

UPDATE Dim_Doctor
SET Specialization_ID = (SELECT Specialization_ID FROM Dim_Specialization WHERE Specialization_Name = 'Neurology')
WHERE First_Name = 'Michael' AND Last_Name = 'Davis';

UPDATE Dim_Doctor
SET Specialization_ID = (SELECT Specialization_ID FROM Dim_Specialization WHERE Specialization_Name = 'Orthopedics')
WHERE First_Name = 'Sarah' AND Last_Name = 'Miller';

UPDATE Dim_Doctor
SET Specialization_ID = (SELECT Specialization_ID FROM Dim_Specialization WHERE Specialization_Name = 'Pediatrics')
WHERE First_Name = 'James' AND Last_Name = 'Wilson';

SELECT Specialization_ID, Specialization_Name 
FROM Dim_Specialization;

SET SQL_SAFE_UPDATES = 1;


UPDATE Dim_Doctor d
JOIN Dim_Specialization s ON s.Specialization_Name = 'Dermatology'
SET d.Specialization_ID = s.Specialization_ID
WHERE d.First_Name = 'Linda' AND d.Last_Name = 'Moore';


-- star schema queries.

SELECT 
    p.First_Name AS Patient_First_Name, 
    p.Last_Name AS Patient_Last_Name, 
    d.First_Name AS Doctor_First_Name, 
    d.Last_Name AS Doctor_Last_Name, 
    s.Service_Type, 
    m.Drug_Name, 
    f.Cost 
FROM 
    Fact_Treatment f
JOIN Dim_Patient p ON f.Patient_ID = p.Patient_ID
JOIN Dim_Doctor d ON f.Doctor_ID = d.Doctor_ID
JOIN Dim_Service s ON f.Service_ID = s.Service_ID
JOIN Dim_Medication m ON f.Medication_ID = m.Medication_ID;


-- snowflake schema

SELECT 
    p.First_Name AS Patient_First_Name, 
    p.Last_Name AS Patient_Last_Name, 
    d.First_Name AS Doctor_First_Name, 
    d.Last_Name AS Doctor_Last_Name, 
    s.Specialization_Name, 
    fa.Appointment_Status 
FROM 
    Fact_Appointment fa
JOIN Dim_Patient p ON fa.Patient_ID = p.Patient_ID
JOIN Dim_Doctor d ON fa.Doctor_ID = d.Doctor_ID
JOIN Dim_Specialization s ON d.Specialization_ID = s.Specialization_ID;


-- galaxy schema

SELECT 
    p.First_Name AS Patient_First_Name, 
    p.Last_Name AS Patient_Last_Name, 
    d.First_Name AS Doctor_First_Name, 
    d.Last_Name AS Doctor_Last_Name, 
    f_t.Cost AS Treatment_Cost, 
    f_a.Appointment_Cost, 
    dd.Full_Date AS Appointment_Date 
FROM 
    Fact_Treatment f_t
JOIN Dim_Patient p ON f_t.Patient_ID = p.Patient_ID
JOIN Dim_Doctor d ON f_t.Doctor_ID = d.Doctor_ID
JOIN Dim_Date dd ON f_t.Date_Key = dd.Date_Key
JOIN Fact_Appointment f_a ON f_a.Patient_ID = p.Patient_ID AND f_a.Doctor_ID = d.Doctor_ID;
