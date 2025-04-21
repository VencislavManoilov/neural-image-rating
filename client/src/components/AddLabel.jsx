import React, { useState, useContext } from 'react';
import { useNavigate, Link } from 'react-router-dom';
import axios from 'axios';
import AuthContext from '../context/AuthContext';
import './addLabel.css';

const URL = process.env.REACT_APP_API_URL || 'http://localhost:8080';

function AddLabel() {
  const [labelName, setLabelName] = useState('');
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [error, setError] = useState(null);
  
  const { user } = useContext(AuthContext);
  const navigate = useNavigate();

  // Redirect if not logged in
  if (!user) {
    navigate('/login');
    return null;
  }

  const handleChange = (e) => {
    setLabelName(e.target.value);
    if (error) setError(null);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    
    if (!labelName.trim()) {
      setError('Label name cannot be empty');
      return;
    }
    
    try {
      setIsSubmitting(true);
      const response = await axios.post(`${URL}/labels/create`, 
        { label: labelName },
        {
          headers: {
            Authorization: `Bearer ${localStorage.getItem('token')}`
          }
        }
      );
      
      if (response.data.success) {
        navigate('/labels');
      } else {
        setError(response.data.message || 'Failed to create label');
      }
    } catch (error) {
      console.error('Error creating label:', error);
      setError(error.response?.data?.message || 'Failed to create label. Please try again.');
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <div className="add-label-container">
      <div className="add-label-card">
        <h2>Create New Label</h2>
        
        {error && <div className="error-message">{error}</div>}
        
        <form onSubmit={handleSubmit}>
          <div className="form-group">
            <label htmlFor="label-name">Label Name</label>
            <input
              type="text"
              id="label-name"
              value={labelName}
              onChange={handleChange}
              placeholder="Enter a name for your label"
              autoFocus
            />
          </div>
          
          <div className="add-label-actions">
            <button 
              type="submit" 
              className="add-label-submit"
              disabled={isSubmitting}
            >
              {isSubmitting ? 'Creating...' : 'Create Label'}
            </button>
            
            <Link to="/labels" className="add-label-cancel">
              Cancel
            </Link>
          </div>
        </form>
      </div>
    </div>
  );
}

export default AddLabel;
