import React, { useState, useEffect } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import axios from 'axios';
import './labels.css';

const URL = process.env.REACT_APP_API_URL || 'http://localhost:8080';

function Labels() {
  const [labels, setLabels] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [editingLabel, setEditingLabel] = useState(null);
  const [editValue, setEditValue] = useState('');
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


  const handleEditStart = (label) => {
    setEditingLabel(label.label);
    setEditValue(label.label);
  };

  const handleEditCancel = () => {
    setEditingLabel(null);
    setEditValue('');
  };

  const handleEditSave = async (name) => {
    if (!editValue.trim()) {
      return;
    }
    
    try {
      const response = await axios.put(`${URL}/labels/update/${name}`, 
        { label: editValue },
        {
          headers: {
            Authorization: `Bearer ${localStorage.getItem('token')}`
          }
        }
      );
      
      if (response.data.success) {
        // Update local state with edited label
        setLabels(labels.map(label => 
          label.label === name ? { ...label, label: editValue } : label
        ));
        setEditingLabel(null);
      }
    } catch (error) {
      console.error('Error updating label:', error);
      setError('Failed to update label. Please try again.');
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
              {editingLabel === label.label ? (
                <div className="label-edit">
                  <input
                    type="text"
                    value={editValue}
                    onChange={(e) => setEditValue(e.target.value)}
                    autoFocus
                    className="label-edit-input"
                  />
                  <div className="label-edit-actions">
                    <button 
                      className="edit-save-button" 
                      onClick={() => handleEditSave(label.label)}
                    >
                      Save
                    </button>
                    <button 
                      className="edit-cancel-button" 
                      onClick={handleEditCancel}
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
                      className="edit-button" 
                      onClick={() => handleEditStart(label)}
                    >
                      Edit
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
