from flask import Flask, request, jsonify
from database import db
from models import Seat, Booking
from rate_limiter import rate_limit
import os

app = Flask(__name__)

# PostgreSQL connection string from environment variables
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URL', 'postgresql://user:password@db:5432/ticketdb')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db.init_app(app)

@app.before_request
def create_tables():
    # In a production app, use Flask-Migrate. This is for quick setup.
    db.create_all()
    # Populate dummy seats if empty
    if not Seat.query.first():
        for i in range(1, 11):
            db.session.add(Seat(seat_number=f"A{i}"))
        db.session.commit()

@app.route('/book', methods=['POST'])
@rate_limit
def book_ticket():
    data = request.get_json()
    target_seat = data.get('seat_number')
    user_id = data.get('user_id')

    if not target_seat or not user_id:
        return jsonify({"error": "Missing seat_number or user_id"}), 400

    try:
        # PESSIMISTIC ROW-LEVEL LOCKING (The most critical part of the project)
        # This locks the specific seat row in PostgreSQL until the transaction completes.
        seat = Seat.query.filter_by(seat_number=target_seat).with_for_update().first()

        if not seat:
            return jsonify({"error": "Seat not found"}), 404

        if seat.is_booked:
            return jsonify({"error": "Seat is already booked"}), 409

        # Process booking
        seat.is_booked = True
        new_booking = Booking(seat_id=seat.id, user_id=user_id)
        db.session.add(new_booking)
        
        # Commit transaction (releases the lock)
        db.session.commit()

        return jsonify({"message": f"Successfully booked {target_seat} for user {user_id}"}), 200

    except Exception as e:
        db.session.rollback() # ACID compliance: revert if anything fails
        return jsonify({"error": "Transaction failed", "details": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
