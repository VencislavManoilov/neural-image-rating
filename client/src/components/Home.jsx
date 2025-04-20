import { useContext } from 'react';
import { Link } from 'react-router-dom';
import AuthContext from '../context/AuthContext';
import './home.css';

function Home() {
  const { user, loading, logout } = useContext(AuthContext);

  if (loading) {
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
            <Link to="/rate" className="action-button primary">
              Start Rating Images
            </Link>
            
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