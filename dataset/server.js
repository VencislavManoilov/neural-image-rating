const express = require('express');
const fs = require('fs');
const path = require('path');
const csv = require('csv-parser');
const { createObjectCsvWriter } = require('csv-writer');
const { version } = require('os');
const cors = require('cors'); // Add CORS package

const app = express();
const PORT = process.env.PORT || 5000;
const IMAGES_DIR = path.join(__dirname, 'images');
const LABELS_FILE = path.join(__dirname, 'labels.csv');

// Enable CORS for the React client
app.use(cors());

// Ensure CSV file exists with headers
if (!fs.existsSync(LABELS_FILE)) {
  fs.writeFileSync(LABELS_FILE, 'image,rating\n');
}

app.get('/', (req, res) => {
    res.status(200).json({
        message: 'Welcome to the Image Rating Dataset API',
        version: "1.0.0",
    });
});

// GET /images - Return all image names
app.get('/images', (req, res) => {
  fs.readdir(IMAGES_DIR, (err, files) => {
    if (err) {
      console.error('Error reading images directory:', err);
      return res.status(500).json({ error: 'Failed to read images directory' });
    }
    
    // Filter for image files (jpg, png, etc.)
    const imageFiles = files.filter(file => 
      /\.(jpg|jpeg|png|gif|bmp|webp)$/i.test(file)
    );
    
    res.json({ images: imageFiles });
  });
});

// GET /get/:name - Serve the image file
app.get('/get/:name', (req, res) => {
  const imageName = req.params.name;
  const imagePath = path.join(IMAGES_DIR, imageName);
  
  // Check if file exists
  fs.access(imagePath, fs.constants.F_OK, (err) => {
    if (err) {
      return res.status(404).json({ error: 'Image not found' });
    }
    
    // Serve the image file
    res.sendFile(imagePath);
  });
});

// GET /labels - Get labels.csv
app.get('/labels', (req, res) => {
  // Check if labels file exists
  fs.access(LABELS_FILE, fs.constants.F_OK, (err) => {
    if (err) {
      return res.status(404).json({ error: 'Labels file not found' });
    }
    
    // Read the CSV file
    const results = [];
    fs.createReadStream(LABELS_FILE)
      .pipe(csv())
      .on('data', (data) => results.push(data))
      .on('end', () => {
        res.json({ labels: results });
      })
      .on('error', (err) => {
        console.error('Error reading labels file:', err);
        res.status(500).json({ error: 'Failed to read labels file' });
      });
  });
});

// POST /rate/:name?rating=... - Save or update image rating
app.post('/rate/:name', (req, res) => {
  const imageName = req.params.name;
  const rating = req.query.rating;
  
  // Validate input
  if (!rating) {
    return res.status(400).json({ error: 'Rating is required' });
  }
  
  // Verify image exists
  const imagePath = path.join(IMAGES_DIR, imageName);
  if (!fs.existsSync(imagePath)) {
    return res.status(404).json({ error: 'Image not found' });
  }

  // Read existing ratings
  const ratings = [];
  let imageExists = false;
  
  // Create a promise to handle CSV reading
  const readCsv = new Promise((resolve, reject) => {
    fs.createReadStream(LABELS_FILE)
      .pipe(csv())
      .on('data', (data) => {
        if (data.image === imageName) {
          // Update existing rating
          data.rating = rating;
          imageExists = true;
        }
        ratings.push(data);
      })
      .on('end', () => {
        if (!imageExists) {
          // Add new rating
          ratings.push({ image: imageName, rating });
        }
        resolve();
      })
      .on('error', (err) => {
        reject(err);
      });
  });

  // Write updated ratings to CSV
  readCsv.then(() => {
    const csvWriter = createObjectCsvWriter({
      path: LABELS_FILE,
      header: [
        { id: 'image', title: 'image' },
        { id: 'rating', title: 'rating' }
      ]
    });
    
    return csvWriter.writeRecords(ratings);
  })
  .then(() => {
    res.json({ 
      success: true, 
      message: imageExists ? 'Rating updated' : 'Rating added',
      image: imageName,
      rating: rating
    });
  })
  .catch((err) => {
    console.error('Error processing ratings:', err);
    res.status(500).json({ error: 'Failed to save rating' });
  });
});

// Start the server
app.listen(PORT, () => {
  console.log(`Server is running on http://localhost:${PORT}`);
});