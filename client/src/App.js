import React from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import { AuthProvider } from './context/AuthContext';
import Home from './components/Home';
import Login from './components/Login';
import Register from './components/Register';
import Rate from './components/Rate';
import Labels from './components/Labels';
import AddLabel from './components/AddLabel';
import LabelDetail from './components/LabelDetail';
import './App.css';

function App() {
  return (
    <AuthProvider>
      <Router>
        <div className="app">
          <Routes>
            <Route path="/" element={<Home />} />
            <Route path="/login" element={<Login />} />
            <Route path="/register" element={<Register />} />
            <Route path="/rate" element={<Rate />} />
            <Route path="/labels" element={<Labels />} />
            <Route path="/add-label" element={<AddLabel />} />
            <Route path="/labels/:id" element={<LabelDetail />} />
            <Route path="*" element={<Navigate to="/" replace />} />
          </Routes>
        </div>
      </Router>
    </AuthProvider>
  );
}

export default App;