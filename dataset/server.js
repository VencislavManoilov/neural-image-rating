const express = require('express');
const fs = require('fs');
const path = require('path');
const csv = require('csv-parser');
const { createObjectCsvWriter } = require('csv-writer');
const cors = require('cors'); // Add CORS package

const app = express();
const PORT = process.env.PORT || 5000;
const IMAGES_DIR = path.join(__dirname, 'images');
const LABELS_DIR = path.join(__dirname, 'labels');

const allowedOrigins = process.env.CORS_ORIGIN ? process.env.CORS_ORIGIN.split(',') : ['http://localhost:8080'];
app.use(cors({
    origin: allowedOrigins,
    methods: ['GET', 'POST', 'PUT', 'DELETE'],
}));

app.use(express.json());
app.use(express.urlencoded({ extended: true }));

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
app.get('/labels/:name', (req, res) => {
  const labelsName = req.params.name;
  const labelsPath = path.join(LABELS_DIR, labelsName);

  // Check if labels file exists
  if(!fs.existsSync(labelsPath)) {
    return res.status(404).json({ error: 'Labels file not found' });
  }

  fs.access(labelsPath, fs.constants.F_OK, (err) => {
    if (err) {
      return res.status(404).json({ error: 'Labels file not found' });
    }
    
    // Read the CSV file
    const results = [];
    fs.createReadStream(labelsPath)
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

// POST /add - Create a new labels file in labels directory with the name provided in the request body
app.post('/add', (req, res) => {
  const { name } = req.body;
  if (!name) {
    return res.status(400).json({ error: 'Name is required' });
  }
  
  const labelsPath = path.join(LABELS_DIR, name);
  
  // Check if file already exists
  if (fs.existsSync(labelsPath)) {
    return res.status(400).json({ error: 'Labels file already exists' });
  }
  
  // Create a new labels file
  try {
    fs.writeFileSync(labelsPath, 'image,rating\n'); // Initialize with headers
    res.status(200).json({ message: 'Labels file created', name });
  } catch (err) {
    console.error('Error creating labels file:', err);
    return res.status(500).json({ error: 'Failed to create labels file' });
  }
});

// DELETE /delete/:name - Delete a labels file
app.delete('/labels/:name', (req, res) => {
  const labelsName = req.params.name;
  const labelsPath = path.join(LABELS_DIR, labelsName);
  
  // Check if file exists
  if (!fs.existsSync(labelsPath)) {
    return res.status(404).json({ error: 'Labels file not found' });
  }
  
  // Delete the file
  fs.unlink(labelsPath, (err) => {
    if (err) {
      console.error('Error deleting labels file:', err);
      return res.status(500).json({ error: 'Failed to delete labels file' });
    }
    
    res.json({ message: 'Labels file deleted' });
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