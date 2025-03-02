# README

## Project Setup

This project consists of two main components:
- **Frontend**: A React TypeScript website managed with `npm`.
- **Backend**: A Python Flask server.

### Prerequisites
Ensure you have the following installed:
- [Node.js](https://nodejs.org/) (includes npm)
- [Python](https://www.python.org/) (>=3.8 recommended)
- [pip](https://pip.pypa.io/en/stable/) (Python package manager)
- [virtualenv](https://virtualenv.pypa.io/en/latest/) (optional but recommended)

---

## Frontend Setup (React TypeScript)

1. Navigate to the `frontend` directory:
   ```sh
   cd webdetective
   ```

2. Install dependencies:
   ```sh
   npm install
   ```

3. Build server:
   ```sh
   npm run build
   ```

4. Preview server:
   ```sh
  npm run preview
   ```
   The React app should now be running at `http://localhost:3000/` (or another available port).


---

## Backend Setup (Flask)

1. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```

2. (Optional) Create and activate a virtual environment:
   ```sh
   python -m venv venv
   source venv/bin/activate  # On macOS/Linux
   venv\Scripts\activate     # On Windows
   ```

3. Navigate to the `backend` directory:
   ```sh
   cd webdetective/backend
   ```


4. Run the Flask server:
   ```sh
   python server.py
   ```
   The Flask server should now be running at `http://localhost:5000/`.

### Additional Commands
- **Run in development mode**:
  ```sh
  FLASK_ENV=development flask run
  ```
- **Run tests**:
  ```sh
  pytest
  ```

---

## Connecting Frontend and Backend
- Ensure the frontend is making API
