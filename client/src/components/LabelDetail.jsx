import React, { useState, useEffect } from 'react';
import { useParams, Link } from 'react-router-dom';
import axios from 'axios';
import './labelDetail.css';

const DATASET_URL = process.env.REACT_APP_DATASET_URL || 'http://localhost:5000';
const URL = process.env.REACT_APP_API_URL || 'http://localhost:8080';

function LabelDetail() {
  const [labels, setLabels] = useState(null);
  const [labelsDetails, setLabelsDetails] = useState({});
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  
  const { id } = useParams();
  
  useEffect(() => {
    const fetchLabelsDetail = async () => {
      try {
        setLoading(true);
        const response = await axios.get(`${URL}/labels/get/${id}`, {
          headers: {
            Authorization: `Bearer ${localStorage.getItem('token')}`
          }
        });
        
        if (response.data.labels) {
          setLabels(response.data.labels);
          setLabelsDetails(response.data.labelsDetails);
        } else {
          setError('Label not found');
        }
        setLoading(false);
      } catch (error) {
        console.error('Error fetching label details:', error);
        setError('Failed to load label details. Please try again.');
        setLoading(false);
      }
    };
    
    fetchLabelsDetail();
  }, [id]);
  
  
  if (loading) {
    return (
      <div className="loading">
        <div className="spinner"></div>
        <h2>Loading label details...</h2>
      </div>
    );
  }
  
  if (error) {
    return (
      <div className="label-detail-container">
        <div className="label-detail-error">
          <h2>Error</h2>
          <p>{error}</p>
          <Link to="/labels" className="back-to-labels">
            Back to Labels
          </Link>
        </div>
      </div>
    );
  }
  
  if (!labels) {
    return (
      <div className="label-detail-container">
        <div className="label-detail-not-found">
          <h2>Label Not Found</h2>
          <Link to="/labels" className="back-to-labels">
            Back to Labels
          </Link>
        </div>
      </div>
    );
  }
  
  return (
    <div className="label-detail-container">
      <div className="label-detail-header">
        <h1>{labels.label}</h1>
        <Link to="/labels" className="back-to-labels">
          Back to Labels
        </Link>
      </div>
      
      <div className="label-detail-card">
        <div className="label-detail-info">
          <div className="detail-item">
            <span className="detail-label">Created:</span>
            <span className="detail-value">
              {new Date(labelsDetails.created_at).toLocaleDateString()} at {new Date(labelsDetails.created_at).toLocaleTimeString()}
            </span>
          </div>
          
          <div className="detail-item">
            <span className="detail-label">Label Name:</span>
            <span className="detail-value">{labelsDetails.label}</span>
          </div>
          
          {labels.updated_at && (
            <div className="detail-item">
              <span className="detail-label">Last Updated:</span>
              <span className="detail-value">
                {new Date(labels.updated_at).toLocaleDateString()} at {new Date(labels.updated_at).toLocaleTimeString()}
              </span>
            </div>
          )}
        </div>
        
        <div className="label-detail-actions">
          <Link to={`/labels/edit/${id}`} className="edit-label-button">
            Edit Label
          </Link>
          <button className="delete-label-button">
            Delete Label
          </button>
        </div>
        
        {labels.length > 0 ? (
          <div className="label-detail-images">
            <h2>Images Associated with this Label</h2>
            <div className="image-list">
              {labels.map((image) => (
                <div key={image.image} className="image-item">
                  <img src={DATASET_URL+"/get/"+image.image} className="image-thumbnail" />
                  <p>{image.rating}</p>
                </div>
              ))}
            </div>
          </div>
        ) : (
            <div className="no-images">
                <p>No images associated with this label yet.</p>
            </div>
        )}
      </div>
    </div>
  );
}

export default LabelDetail;
