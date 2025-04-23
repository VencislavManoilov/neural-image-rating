import React, { useState, useEffect } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import axios from 'axios';
import './labels.css';

const URL = process.env.REACT_APP_API_URL || 'http://localhost:8080';

function Labels() {
  const [labels, setLabels] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [trainConfirm, setTrainConfirm] = useState(null);
  const [deleteConfirm, setDeleteConfirm] = useState(null);

  const navigate = useNavigate();

  useEffect(() => {
    const fetchLabels = async () => {
      try {
        setLoading(true);
        const response = await axios.get(`${URL}/labels/all`, {
          headers: {
            Authorization: `Bearer ${localStorage.getItem('token')}`
          }
        });
  
        if (response.data.labels) {
          setLabels(response.data.labels);
        }
        setLoading(false);
      } catch (error) {
        console.error('Error fetching labels:', error);
        setError('Failed to load labels. Please try again later.');
        setLoading(false);
      }
    };

    fetchLabels();
  }, []);
  
  const handleTrainConfirm = (name) => {
    setTrainConfirm(name);
  };

  const handleTrainCancel = () => {
    setTrainConfirm(null);
  };

  const handleTrain = async (label) => {
    try {
      const response = await axios.post(`${URL}/labels/train/${label}`, {}, {
        headers: {
          Authorization: `Bearer ${localStorage.getItem('token')}`
        }
      });
      
      if (response.status === 200) {
        setTrainConfirm(null);
        alert(`Training started successfully for label: ${label}`);
      }
    } catch (error) {
      console.error('Error starting training:', error);
      setError('Failed to start training. Please try again.');
      setTrainConfirm(null);
    }
  };

  const handleDeleteConfirm = (name) => {
    setDeleteConfirm(name);
  };

  const handleDeleteCancel = () => {
    setDeleteConfirm(null);
  };

  const handleDelete = async (name) => {
    try {
      await axios.delete(`${URL}/labels/delete/${name}`, {
        headers: {
          Authorization: `Bearer ${localStorage.getItem('token')}`
        }
      });
      
      window.location.reload();
    } catch (error) {
      console.error('Error deleting label:', error);
      setError('Failed to delete label. Please try again.');
      setDeleteConfirm(null);
    }
  };

  const handleAddNewLabel = async () => {
    navigate('/add-label');
  };

  if (loading) {
    return (
      <div className="loading">
        <div className="spinner"></div>
        <h2>Loading labels...</h2>
      </div>
    );
  }
  
  return (
    <div className="labels-container">
      <div className="labels-header">
        <h1>Your Labels</h1>
        <Link to="/" className="back-button">
          Back to Home
        </Link>
      </div>

      {error && <div className="error-message">{error}</div>}

      <div className="labels-actions">
        <button className="add-label-button" onClick={handleAddNewLabel}>
          <span className="plus-icon">+</span> Add New Label
        </button>
      </div>

      {labels.length === 0 ? (
        <div className="no-labels">
          <p>You haven't created any labels yet.</p>
          <button onClick={handleAddNewLabel} className="action-button primary">
            Create Your First Label
          </button>
        </div>
      ) : (
        <div className="labels-list">
          {labels.map(label => (
            <div key={label.label} className="label-card">
              {trainConfirm === label.label ? (
                <div className="label-train-confirm">
                  <p>Are you sure you want to train this label?</p>
                  <div className="train-confirm-actions">
                    <button 
                      className="train-confirm-button" 
                      onClick={() => handleTrain(label)}
                    >
                      Yes, Train
                    </button>
                    <button 
                      className="train-cancel-button" 
                      onClick={handleTrainCancel}
                    >
                      Cancel
                    </button>
                  </div>
                </div>
              ) : deleteConfirm === label.label ? (
                <div className="label-delete-confirm">
                  <p>Are you sure you want to delete this label?</p>
                  <div className="delete-confirm-actions">
                    <button 
                      className="delete-confirm-button" 
                      onClick={() => handleDelete(label.label)}
                    >
                      Yes, Delete
                    </button>
                    <button 
                      className="delete-cancel-button" 
                      onClick={handleDeleteCancel}
                    >
                      Cancel
                    </button>
                  </div>
                </div>
              ) : (
                <>
                  <div className="label-info">
                    <h3>{label.label}</h3>
                    <p className="label-date">Created on: {new Date(label.created_at).toLocaleDateString()}</p>
                  </div>
                  <div className="label-actions">
                    <Link to={`/labels/${label.label}`} className="view-button">
                      View
                    </Link>
                    <button 
                      className="train-button" 
                      onClick={() => handleTrainConfirm(label.label)}
                    >
                      Train
                    </button>
                    <button 
                      className="delete-button" 
                      onClick={() => handleDeleteConfirm(label.label)}
                    >
                      Delete
                    </button>
                  </div>
                </>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

export default Labels;
