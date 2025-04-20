const knex = require('./knex');
const logger = require('./utils/logger');

const schemaDefinition = {
  users: {
    id: { type: 'increments', primary: true },
    username: { type: 'string', notNullable: true },
    email: { type: 'string', unique: true, notNullable: true },
    password: { type: 'string', notNullable: true },
    role: { type: 'string', defaultTo: 'user' },
    created_at: { type: 'timestamp', defaultTo: knex.fn.now() },
  },
  labels: {
    id: { type: 'increments', primary: true },
    userId: { type: 'integer', notNullable: true },
    label: { type: 'string', notNullable: true },
    created_at: { type: 'timestamp', defaultTo: knex.fn.now() },
  }
};

async function ensureSchema() {
  for (const [tableName, columns] of Object.entries(schemaDefinition)) {
    const exists = await knex.schema.hasTable(tableName);

    if (!exists) {
      logger.info(`Creating ${tableName} table...`);
      await knex.schema.createTable(tableName, (table) => {
        for (const [columnName, columnProps] of Object.entries(columns)) {
          let column;
          if (columnProps.type === 'increments') {
            column = table.increments(columnName);
          } else {
            column = table[columnProps.type](columnName);
          }

          if (columnProps.primary) column.primary();
          if (columnProps.notNullable) column.notNullable();
          if (columnProps.unique) column.unique();
          if (columnProps.defaultTo) column.defaultTo(columnProps.defaultTo);
        }
      });
    } else {
      logger.info(`Checking for missing columns in ${tableName} table...`);
      const existingColumns = await knex(tableName).columnInfo();

      for (const [columnName, columnProps] of Object.entries(columns)) {
        if (!existingColumns[columnName]) {
          logger.info(`Adding "${columnName}" column to ${tableName} table...`);
          await knex.schema.alterTable(tableName, (table) => {
            let column;
            if (columnProps.type === 'increments') {
              column = table.increments(columnName);
            } else {
              column = table[columnProps.type](columnName);
            }

            if (columnProps.primary) column.primary();
            if (columnProps.notNullable) column.notNullable();
            if (columnProps.unique) column.unique();
            if (columnProps.defaultTo) column.defaultTo(columnProps.defaultTo);
          });
        }
      }
    }
  }

  logger.info('Schema check complete.');
}

module.exports = ensureSchema;