import { useState, useEffect } from 'react';
import axios from 'axios';
import './rate.css';

const DATASET_URL = process.env.REACT_APP_DATASET_URL || 'http://localhost:5000';
const URL = process.env.REACT_APP_API_URL || 'http://localhost:8080';

function Rate({ label }) {
  const [images, setImages] = useState([]);
  const [currentIndex, setCurrentIndex] = useState(0);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);

  // Fetch the list of image names when the component mounts
  useEffect(() => {
    const fetchImages = async () => {
      try {
        setIsLoading(true);
        const response = await fetch(DATASET_URL+'/images');
        
        if (!response.ok) {
          throw new Error(`HTTP error! Status: ${response.status}`);
        }
        
        const data = await response.json();
        setImages(data.images);
        setIsLoading(false);
      } catch (error) {
        setError('Failed to fetch images. Please try again later.');
        setIsLoading(false);
        console.error('Error fetching images:', error);
      }
    };

    if(label) {
      fetchImages();
    }

    console.log('Label:', label);
  }, [label]);

  // Handle rating submission
  const handleRating = async (rating) => {
    if (images.length === 0 || currentIndex >= images.length) return;
    
    const currentImage = images[currentIndex];
    
    try {
      await axios.post(`${URL}/labels/rate/${label?.name}/${currentImage}?rating=${rating}`, {}, {
        headers: {
          Authorization: `Bearer ${localStorage.getItem('token')}`,
        }
      });

      const image = label.labels.find(l => l.image === currentImage);

      if(image && image.rating) {
        image.rating = rating;
      } else {
        label.labels.push({ image: currentImage, rating });
      }
      
      // Move to the next image
      if (currentIndex < images.length - 1) {
        setCurrentIndex(currentIndex + 1);
      } else {
        // If at the last image, show completion message or loop back
        alert('You have rated all images!');
        setCurrentIndex(0);
      }
    } catch (error) {
      console.error('Error submitting rating:', error);
      setError('Failed to submit rating. Please try again.');
    }
  };

  // Navigate to previous image
  const handlePrevious = () => {
    if (currentIndex > 0) {
      setCurrentIndex(currentIndex - 1);
    }
  };

  // Navigate to next image
  const handleNext = () => {
    if (currentIndex < images.length - 1) {
      setCurrentIndex(currentIndex + 1);
    }
  };

  // Create array of rating buttons from 1-10
  const ratingButtons = (rating) => {
    return [...Array(10)].map((_, i) => (
      <button
        key={i + 1}
        className={"rating-button" + (parseInt(rating) === i + 1 ? ' selected' : '')}
        onClick={() => handleRating(i + 1)}
      >
        {i + 1}
      </button>
    ));
  };

  if (isLoading) {
    return <div className="loading">Loading images...</div>;
  }

  if (error) {
    return <div className="error">{error}</div>;
  }

  return (
    <div className="rate-container">
      <div className="image-viewer">
        <h3 className="index">{currentIndex}/{images.length}</h3>
        {images.length > 0 ? (
          <>
            <div className="navigation">
              <button 
                className="nav-button prev"
                onClick={handlePrevious}
                disabled={currentIndex === 0}
              >
                &#8592;
              </button>
              <div className="image-container">
                <img 
                  src={`${DATASET_URL}/get/${images[currentIndex]}`}
                  alt=""
                />
              </div>
              <button 
                className="nav-button next"
                onClick={handleNext}
                disabled={currentIndex === images.length - 1}
              >
                &#8594;
              </button>
            </div>
            <div className="rating-container">
              <p>Rate this image:</p>
              <div className="rating-buttons">
                {ratingButtons(label?.labels.find(l => l.image === images[currentIndex])?.rating)}
              </div>
            </div>
          </>
        ) : (
          <div className="no-images">No images found</div>
        )}
      </div>
    </div>
  );
}

export default Rate
