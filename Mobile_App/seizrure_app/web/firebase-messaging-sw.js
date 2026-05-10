importScripts("https://www.gstatic.com/firebasejs/9.10.0/firebase-app-compat.js");
importScripts("https://www.gstatic.com/firebasejs/9.10.0/firebase-messaging-compat.js");

firebase.initializeApp({
  apiKey: "AIzaSyDJIrg7FEiBiasnA2hbhzrB_Bk_NjCqWSM",
  authDomain: "heart-rate-879b0.firebaseapp.com",
  databaseURL: "https://heart-rate-879b0-default-rtdb.firebaseio.com",
  projectId: "heart-rate-879b0",
  storageBucket: "heart-rate-879b0.firebasestorage.app",
  messagingSenderId: "207018206449",
  appId: "1:207018206449:web:3959687d5107f31a017550"
});

const messaging = firebase.messaging();

// Optional: Handle background messages
messaging.onBackgroundMessage((payload) => {
  console.log("Received background message: ", payload);
  const notificationTitle = payload.notification.title;
  const notificationOptions = {
    body: payload.notification.body,
    icon: "/favicon.png"
  };

  return self.registration.showNotification(
    notificationTitle,
    notificationOptions
  );
});
