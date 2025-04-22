const express = require('express');
const router = express.Router();
const bcrypt = require('bcrypt');
const jwt = require('jsonwebtoken');
const knex = require('../knex');
const Authorization = require('../middleware/Authorization');
const logger = require('../utils/logger');

// JWT secret key - in production this should be in environment variables
const JWT_SECRET = process.env.JWT_SECRET || 'your-jwt-secret-key';

// Register endpoint
router.post('/register', async (req, res) => {
  try {
    const { username, email, password } = req.body;
    
    // Validate input
    if (!username || !email || !password) {
      return res.status(400).json({ message: 'All fields are required' });
    }
    
    // Check if user already exists
    const existingUser = await knex('users').where({ email }).first();
    if (existingUser) {
      return res.status(409).json({ message: 'User already exists' });
    }
    
    // Hash password
    const salt = await bcrypt.genSalt(10);
    const hashedPassword = await bcrypt.hash(password, salt);
    
    // Insert user into database
    const [newUser] = await knex('users')
      .insert({
        username,
        email,
        password: hashedPassword
      })
      .returning(['id', 'username', 'email', 'role']);
      
    // Generate JWT token
    const token = jwt.sign({ email: newUser.email }, JWT_SECRET, { expiresIn: '7d' });
    
    res.status(200).json({ 
      message: 'User registered successfully',
      user: {
        id: newUser.id,
        username: newUser.username,
        email: newUser.email,
        role: newUser.role
      },
      token
    });
    
  } catch (error) {
    logger.error('Registration error:' + error);
    res.status(500).json({ message: 'Server error during registration' });
  }
});

// Login endpoint
router.post('/login', async (req, res) => {
  try {
    const { email, password } = req.body;
    
    // Validate input
    if (!email || !password) {
      return res.status(400).json({ message: 'Email and password are required' });
    }
    
    // Find user
    const user = await knex('users').where({ email }).first();
    if (!user) {
      return res.status(400).json({ message: 'Invalid credentials' });
    }
    
    // Verify password
    const validPassword = await bcrypt.compare(password, user.password);
    if (!validPassword) {
      return res.status(400).json({ message: 'Invalid credentials' });
    }
    
    // Generate JWT token
    const token = jwt.sign({ email: user.email }, JWT_SECRET, { expiresIn: '7d' });
    
    res.json({ 
      message: 'Login successful',
      user: {
        id: user.id,
        username: user.username,
        email: user.email,
        role: user.role
      },
      token
    });
    
  } catch (error) {
    logger.error('Login error:' + error);
    res.status(500).json({ message: 'Server error during login' });
  }
});

// Me endpoint - Get current user info
router.get('/me', Authorization, async (req, res) => {
  try {
    // User is already attached to request by Authorization middleware
    const user = req.user;
    
    res.json({
      id: user.id,
      username: user.username,
      email: user.email,
      role: user.role
    });
    
  } catch (error) {
    logger.error('Me endpoint error:' + error);
    res.status(500).json({ message: 'Server error' });
  }
});

module.exports = router;
