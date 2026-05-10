import 'package:firebase_core/firebase_core.dart' show FirebaseOptions;
import 'package:flutter/foundation.dart'
    show defaultTargetPlatform, kIsWeb, TargetPlatform;

class DefaultFirebaseOptions {
  static FirebaseOptions get currentPlatform {
    if (kIsWeb) {
      return web;
    }
    switch (defaultTargetPlatform) {
      case TargetPlatform.android:
        return android;
      case TargetPlatform.iOS:
        return ios;
      case TargetPlatform.macOS:
        return macos;
      case TargetPlatform.windows:
        return windows;
      case TargetPlatform.linux:
        throw UnsupportedError(
          'DefaultFirebaseOptions have not been configured for linux - '
          'you can reconfigure this by running the FlutterFire CLI again.',
        );
      default:
        throw UnsupportedError(
          'DefaultFirebaseOptions are not supported for this platform.',
        );
    }
  }

  static const FirebaseOptions web = FirebaseOptions(
    apiKey: 'AIzaSyDJIrg7FEiBiasnA2hbhzrB_Bk_NjCqWSM',
    appId: '1:207018206449:web:f9c3754396025d0a017550',
    messagingSenderId: '207018206449',
    projectId: 'heart-rate-879b0',
    authDomain: 'heart-rate-879b0.firebaseapp.com',
    databaseURL: 'https://heart-rate-879b0-default-rtdb.firebaseio.com',
    storageBucket: 'heart-rate-879b0.firebasestorage.app',
  );

  static const FirebaseOptions android = FirebaseOptions(
    apiKey: 'AIzaSyA5JMsYi8XXqGfDZym4cX1Ixz7_Cw_gvpI',
    appId: '1:207018206449:android:a2de0cda4f3f9cd1017550',
    messagingSenderId: '207018206449',
    projectId: 'heart-rate-879b0',
    databaseURL: 'https://heart-rate-879b0-default-rtdb.firebaseio.com',
    storageBucket: 'heart-rate-879b0.firebasestorage.app',
  );

  static const FirebaseOptions ios = FirebaseOptions(
    apiKey: 'AIzaSyD_f5jk8_VXCfYUY9kzOF3aE95EdpqI2YI',
    appId: '1:207018206449:ios:37535a78d6d20b2e017550',
    messagingSenderId: '207018206449',
    projectId: 'heart-rate-879b0',
    databaseURL: 'https://heart-rate-879b0-default-rtdb.firebaseio.com',
    storageBucket: 'heart-rate-879b0.firebasestorage.app',
    iosBundleId: 'com.example.adminApp',
  );

  static const FirebaseOptions macos = FirebaseOptions(
    apiKey: 'AIzaSyD_f5jk8_VXCfYUY9kzOF3aE95EdpqI2YI',
    appId: '1:207018206449:ios:37535a78d6d20b2e017550',
    messagingSenderId: '207018206449',
    projectId: 'heart-rate-879b0',
    databaseURL: 'https://heart-rate-879b0-default-rtdb.firebaseio.com',
    storageBucket: 'heart-rate-879b0.firebasestorage.app',
    iosBundleId: 'com.example.adminApp',
  );

  static const FirebaseOptions windows = FirebaseOptions(
    apiKey: 'AIzaSyDJIrg7FEiBiasnA2hbhzrB_Bk_NjCqWSM',
    appId: '1:207018206449:web:3959687d5107f31a017550',
    messagingSenderId: '207018206449',
    projectId: 'heart-rate-879b0',
    authDomain: 'heart-rate-879b0.firebaseapp.com',
    databaseURL: 'https://heart-rate-879b0-default-rtdb.firebaseio.com',
    storageBucket: 'heart-rate-879b0.firebasestorage.app',
  );
}
