const express = require('express');
const router = express.Router();
const knex = require('../knex');
const logger = require('../utils/logger');
const { default: axios } = require('axios');
const { v4: uuidv4 } = require('uuid');

const DATASET_URL = process.env.DATASET_URL || 'http://localhost:5000';

router.get('/me', async (req, res) => {
    try {
        const userId = req.user.id;
    
        const labels = await knex('labels')
        .where({ userId })
        .first();
    
        if(!labels) {
            return res.status(404).json({ message: 'Labels not found' });
        }
    
        res.json({
            message: 'Labels retrieved successfully',
            labels
        });
    } catch (error) {
        logger.error('Error retrieving labels:' + error);
        res.status(500).json({ message: 'Server error while retrieving labels' });
    }
});

router.get('/all', async (req, res) => {
    try {
        const labels = await knex('labels')
            .select('label', 'created_at')
            .where({ userId: req.user.id });

        if (labels.length === 0) {
            return res.status(404).json({ message: 'No labels found for this user' });
        }

        res.json({
            message: 'Labels retrieved successfully',
            labels
        });
    } catch (error) {
        logger.error('Error retrieving labels:' + error);
        res.status(500).json({ message: 'Server error while retrieving labels' });
    }
});

router.get('/get/:name', async (req, res) => {
    try {
        const labelsName = req.params.name;

        // Check if the labels exist in the dataset
        const labelsCheck = await knex('labels')
            .where({ label: labelsName })
            .first();

        if (!labelsCheck) {
            return res.status(404).json({ message: 'Labels not found in the database' });
        }

        // Check if the labels belong to the user
        if (labelsCheck.userId !== req.user.id) {
            return res.status(403).json({ message: 'Unauthorized access to labels' });
        }

        const response = await axios.get(`${DATASET_URL}/labels/${labelsName}`);
        if (response.status !== 200) {
            return res.status(404).json({ message: 'Labels not found' });
        }

        const { labels } = response.data;

        res.json({
            message: 'Labels retrieved successfully',
            labelsDetails: labelsCheck,
            labels
        });
    } catch (error) {
        logger.error('Error retrieving labels:' + error);
        res.status(500).json({ message: 'Server error while retrieving labels' });
    }
});

router.post('/add', async (req, res) => {
    const name = uuidv4();
    try {
        await knex('labels').insert({
            userId: req.user.id,
            label: name
        });

        const response = await axios.post(`${DATASET_URL}/add`, {
            name
        });

        if (response.status !== 200) {
            return res.status(400).json({ message: 'Failed to add labels' });
        }

        res.status(200).json({
            message: 'Labels added successfully',
            name: response.data.name
        });
    } catch (error) {
        // Remove any entry that might have been created in the database
        try {
            await knex('labels').where({ userId: req.user.id }).delete();
        } catch (dbCleanupError) {
            logger.error('Error cleaning up database after failure:' + dbCleanupError);
        }

        logger.error('Error adding labels:' + error);
        res.status(500).json({ message: 'Server error while adding labels' });
    }
});

router.delete('/delete/:name', async (req, res) => {
    try {
        const labelName = req.params.name;

        // Check if the label exists
        const label = await knex('labels')
            .where({ label: labelName, userId: req.user.id })
            .first();

        if (!label) {
            return res.status(404).json({ message: 'Label not found' });
        }

        // Check if the label belongs to the user
        if (label.userId !== req.user.id) {
            return res.status(403).json({ message: 'Unauthorized access to label' });
        }

        // Delete the label from the database
        await knex('labels').where({ label: labelName }).delete();

        // Send a request to the dataset to delete the label
        const response = await axios.delete(`${DATASET_URL}/labels/${label.label}`);
        if (response.status !== 200) {
            return res.status(400).json({ message: 'Failed to delete labels' });
        }

        res.json({
            message: 'Labels deleted successfully'
        });
    } catch (error) {
        logger.error('Error deleting labels:' + error);
        res.status(500).json({ message: 'Server error while deleting labels' });
    }
});

module.exports = router;