import { useContext, useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import AuthContext from '../context/AuthContext';
import axios from 'axios';
import './home.css';

const URL = process.env.REACT_APP_API_URL || 'http://localhost:8080';

function Home() {
  const { user, loading, logout } = useContext(AuthContext);
  const [labels, setLabels] = useState(false);

  useEffect(() => {
    const checkLabels = async () => {
      try {
        const response = await axios.get(URL+'/labels/all', {
          headers: {
            Authorization: `Bearer ${localStorage.getItem('token')}`
          }
        });
        
        if(response.data.labels && response.data.labels.length > 0) {
          setLabels(true);
        }
      } catch (error) {
        console.error('Error checking labels:', error);
      }
    }

    checkLabels();
  }, []);

  const addLabels = async () => {
    try {
      const response = await axios.post(URL+'/labels/add', {}, {
        headers: {
          Authorization: `Bearer ${localStorage.getItem('token')}`
        }
      });

      if(response.data.name) {
        setLabels(true);
      }
    }
    catch (error) {
      console.error('Error adding labels:', error);
    }
  };

  if(loading) {
    return (
      <div className="loading">
        <div className="spinner"></div>
        <h2>Loading...</h2>
      </div>
    );
  }

  return (
    <div className="home-container">
      {user ? (
        <div className="welcome-section">
          <h1>Welcome to the Image Rating Dataset</h1>
          <p>Hello, <span className="username">{user.username}</span>! You are now logged in.</p>
          
          <div className="action-buttons">
            {labels ? (
              <Link to="/labels" className="action-button">
                View Your Labels
              </Link>
            ) : (
              <>
                <p style={{ marginBottom: "12px" }}>You have not added any labels yet.</p>
                <button className="action-button primary" onClick={addLabels}>
                  Add Labels
                </button>
              </>
            )}
            
            <button onClick={logout} className="action-button secondary">
              Logout
            </button>
          </div>
        </div>
      ) : (
        <div className="welcome-section">
          <h1>Image Rating Dataset</h1>
          <p>Please login or register to rate images</p>
          
          <div className="auth-options">
            <Link to="/login" className="auth-option">
              Login
            </Link>
            <Link to="/register" className="auth-option">
              Register
            </Link>
          </div>
        </div>
      )}
    </div>
  );
}

export default Home;