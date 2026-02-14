# Database Storage Location

## Default Configuration

- **Database Name:** `tsundere_chat`
- **Collection:** `chat_messages`
- **Connection:** `mongodb://localhost:27017` (local MongoDB)

## Local MongoDB Storage

If you're using **local MongoDB**, data is stored on your filesystem:

### Windows (Default Locations)

1. **MongoDB as Windows Service:**
   - Default: `C:\Program Files\MongoDB\Server\{version}\data\`
   - Or: `C:\data\db\`

2. **Custom Installation:**
   - Check your MongoDB configuration file (usually `mongod.cfg`)
   - Look for `storage.dbPath` setting

### Find Your MongoDB Data Directory

```powershell
# Check MongoDB service configuration
Get-Content "C:\Program Files\MongoDB\Server\{version}\bin\mongod.cfg" | Select-String "dbPath"

# Or check running MongoDB process
Get-WmiObject Win32_Process | Where-Object {$_.Name -eq "mongod.exe"} | Select-Object CommandLine
```

### View Database Files

The database files are stored as:
- `tsundere_chat.0`, `tsundere_chat.1`, etc. (data files)
- `tsundere_chat.ns` (namespace file)
- `journal/` (write-ahead log)

## Cloud MongoDB (MongoDB Atlas)

If you're using **MongoDB Atlas** (cloud), data is stored in the cloud:

1. **Update `Backend/.env`:**
   ```env
   MONGO_URI=mongodb+srv://username:password@cluster.mongodb.net/tsundere_chat
   MONGO_DB=tsundere_chat
   ```

2. **Data Location:** Managed by MongoDB Atlas (various regions)

## Database Schema

### Collection: `chat_messages`

Each document contains:

```json
{
  "_id": "ObjectId",
  "username": "string",
  "role": "user | ai",
  "message": "string",
  "emotion_score": "number",
  "emotion_label": "string",
  "timestamp": "ISODate"
}
```

### Indexes

- `(username, timestamp)` - For history queries
- `(username, timestamp DESC)` - For latest state queries

## Accessing the Database

### Using MongoDB Shell

```powershell
# Connect to local MongoDB
mongosh mongodb://localhost:27017

# Use the database
use tsundere_chat

# View all messages
db.chat_messages.find().pretty()

# Count messages for a user
db.chat_messages.countDocuments({username: "touch"})

# View latest messages
db.chat_messages.find().sort({timestamp: -1}).limit(10)
```

### Using MongoDB Compass (GUI)

1. Download: https://www.mongodb.com/products/compass
2. Connect to: `mongodb://localhost:27017`
3. Select database: `tsundere_chat`
4. View collection: `chat_messages`

## Backup & Export

### Export Data

```powershell
# Export entire database
mongodump --uri="mongodb://localhost:27017" --db=tsundere_chat --out=./backup

# Export specific collection
mongoexport --uri="mongodb://localhost:27017" --db=tsundere_chat --collection=chat_messages --out=chat_messages.json
```

### Import Data

```powershell
# Import database
mongorestore --uri="mongodb://localhost:27017" ./backup

# Import collection
mongoimport --uri="mongodb://localhost:27017" --db=tsundere_chat --collection=chat_messages --file=chat_messages.json
```

## Changing Storage Location

### Local MongoDB

1. **Stop MongoDB service:**
   ```powershell
   Stop-Service MongoDB
   ```

2. **Edit MongoDB config** (`mongod.cfg`):
   ```yaml
   storage:
     dbPath: "D:\MongoDB\Data"
   ```

3. **Move existing data** (if any) to new location

4. **Start MongoDB service:**
   ```powershell
   Start-Service MongoDB
   ```

### Cloud MongoDB

Just update `MONGO_URI` in `Backend/.env` to point to your Atlas cluster.
