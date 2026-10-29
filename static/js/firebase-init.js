// Firebase Configuration & Initialization
// Farmer Guide AI - Smart Farming Assistant

const firebaseConfig = {
  apiKey: "AIzaSyAPrQV91W7uG3dlDD2RX97ZzpmFVhCoelE",
  authDomain: "farmer---guide---ai.firebaseapp.com",
  projectId: "farmer---guide---ai",
  storageBucket: "farmer---guide---ai.firebasestorage.app",
  messagingSenderId: "347894762938",
  appId: "1:347894762938:web:521b91eb0b5b335e916183",
  measurementId: "G-X1R931PW5M"
};

// Initialize Firebase App if not already initialized
let firebaseApp;
if (!firebase.apps.length) {
  firebaseApp = firebase.initializeApp(firebaseConfig);
} else {
  firebaseApp = firebase.app();
}

const auth = firebase.auth();
// Set persistent session across browser restarts
auth.setPersistence(firebase.auth.Auth.Persistence.LOCAL).catch(function(err) {
  console.warn("Firebase persistence error:", err);
});

/**
 * Syncs the active Firebase ID token with the Flask backend session.
 * Returns a Promise that resolves when the session cookie is set.
 */
async function syncSessionWithBackend(idToken) {
  try {
    const res = await fetch("/api/auth/session-login", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ id_token: idToken })
    });
    return await res.json();
  } catch (err) {
    console.error("Session sync error:", err);
    return { success: false, error: err.message };
  }
}

/**
 * Clears the Flask backend session on logout.
 */
async function clearBackendSession() {
  try {
    await fetch("/api/auth/session-logout", {
      method: "POST",
      headers: { "Content-Type": "application/json" }
    });
  } catch (err) {
    console.error("Logout sync error:", err);
  }
}

/**
 * Handles Firebase sign out across both frontend and backend.
 */
async function handleFirebaseLogout() {
  try {
    await auth.signOut();
  } catch (e) {
    console.error("Firebase signOut error:", e);
  }
  await clearBackendSession();
  window.location.replace("/login");
}

// Global exposure for templates and inline scripts
window.FarmerAuth = {
  auth: auth,
  syncSessionWithBackend: syncSessionWithBackend,
  clearBackendSession: clearBackendSession,
  handleFirebaseLogout: handleFirebaseLogout
};
