# Firebase Architecture & Data Structure

This document outlines the organization of Firebase services used in the Seizure Monitoring System, including Firestore, Realtime Database (RTDB), and Authentication.

---

## 🔐 Authentication Strategy

The system uses **Firebase Authentication** for identity management. There are three levels of users:

| Role | Creation | Data Linkage |
| :--- | :--- | :--- |
| **Admin** | Manual / Root | Accessed via `AuthController` in the Admin App. |
| **Patient** | Created by Admin | Auth `UID` matches the Firestore `patients/{id}` document ID. |
| **Caregiver** | Created by Admin | Auth `UID` matches the Firestore `caregivers/{id}` document ID. |

> [!NOTE]
> Patient and Caregiver accounts are created via a **Secondary Firebase App** instance in the Admin app to allow admins to register users without disrupting their own login session.

---

## 📑 Cloud Firestore (Relational Data)

Firestore is used for persistent records that require querying and sorting.

### 🧩 `patients` (Collection)
Primary container for all user-specific data.
- **Document ID**: Matches Patient's Auth `UID`.
- **Fields**:
  - `name`: `String`
  - `email`: `String`
  - `phone`: `String`
  - `birthdate`: `Timestamp`
  - `details`: `String` (Medical history/notes)
  - `authUid`: `String`
  - `createdAt`: `Timestamp`
  - `updatedAt`: `Timestamp`

#### 📂 `caregivers` (Subcollection)
*Path: `patients/{patientId}/caregivers/{caregiverId}`*
- **Document ID**: Matches Caregiver's Auth `UID`.
- **Fields**:
  - `name`: `String`
  - `email`: `String`
  - `phone`: `String`
  - `authUid`: `String`
  - `createdAt`: `Timestamp`
  - `updatedAt`: `Timestamp`

#### 📂 `alerts` (Subcollection)
*Path: `patients/{patientId}/alerts/{alertId}`*
- **Fields**:
  - `time`: `Timestamp` (Actual time of the seizure event)
  - `heartRate`: `Number` (Captured at time of alert)
  - `hrv`: `Number` (Heart Rate Variability)
  - `createdAt`: `FieldValue.serverTimestamp()`

---

## ⚡ Realtime Database (Live Monitoring)

RTDB is used for high-frequency updates and live state tracking where sub-second latency is required.

### 📡 `live` (Root Node)
Contains the current stream of data from the patient's wearable device.
- **Node ID**: `{patientId}` (Matches Firestore/Auth UID)
- **Structure**:
  ```json
  {
    "hr": 75,
    "hrv": 42,
    "medication": "None",
    "symptoms": "Aura",
    "sleep": "good",
    "stress": "medium",
    "updatedAt": 1712760000000
  }
  ```
  - `hr` / `hrv`: Raw numerical data.
  - `sleep`: Valid values are `"good"` or `"bad"`.
  - `updatedAt`: UNIX timestamp for staleness checking.

### ⚙️ `meta` (Root Node)
System-wide configuration and metadata.
- `schemaVersion`: `Integer`
- `updatedAt`: `ServerValue.timestamp`

---

## 🔍 Implementation Notes

### Collection Group Queries
The Admin Dashboard uses Firestore **Collection Group Queries** to aggregate data across all patients simultaneously:
- `fs.collectionGroup('alerts')`: Fetches recent alerts globally for the "Latest Alerts" widget.
- `fs.collectionGroup('caregivers')`: used to calculate the total number of caregivers in the system.

### Data Consistency
Since Patient Auth users and Firestore documents share the same `UID`, the app uses `patientsCol.doc(uid)` for all lookups, ensuring a 1:1 mapping between identity and record.
