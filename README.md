# Epilepsy ECG Project

This project is a seizure risk prediction system based on ECG-related health indicators.

The system includes four main parts:

## 1. Mobile Application
The mobile application is used by the patient and caregiver to monitor health status, enter health factors, and receive seizure risk alerts.

## 2. Smartwatch Application
The smartwatch interface displays the patient's seizure risk status and changes visually when a high-risk alert is detected.

## 3. Admin Dashboard
The admin dashboard is used to manage and monitor users and project-related data.

## 4. Machine Learning Model
The machine learning model predicts seizure risk using health indicators such as HR, HRV, medication status, symptoms, sleep quality, and stress level.

The model was trained using Random Forest. Instead of using TensorFlow Lite, the trained model was converted into native Dart code using m2cgen to make it easier to integrate directly with the Flutter application.

## Repository Structure

```text
Epilepsy-ECG-proj
├── Mobile_App
├── Watch_Code
├── Admin_Code
├── Model_Code
└── README.md