from app import create_app

app = create_app()

if __name__ == "__main__":
    # Start the Flask app
    app.run(host="0.0.0.0", port=5000, debug=app.config["BOOKMIND_SETTINGS"].app_env == "development")
