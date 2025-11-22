#!/bin/bash

# Base URL
BASE_URL="http://localhost:3000/api/auth"

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Helper function to print status
print_status() {
  if [ $1 -eq 0 ]; then
    echo -e "${GREEN}SUCCESS${NC}: $2"
  else
    echo -e "${RED}FAILED${NC}: $2"
    exit 1
  fi
}

# 1. Register a new user
echo "1. Registering new user..."
EMAIL="testuser_$(date +%s)@example.com"
PASSWORD="Password123"
USERNAME="user_$(date +%s)"

REGISTER_RESPONSE=$(curl -s -X POST "$BASE_URL/register" \
  -H "Content-Type: application/json" \
  -d "{\"email\": \"$EMAIL\", \"password\": \"$PASSWORD\", \"confirmPassword\": \"$PASSWORD\", \"username\": \"$USERNAME\"}")

echo "Response: $REGISTER_RESPONSE"

if [[ $REGISTER_RESPONSE == *"success\":true"* ]]; then
  print_status 0 "User registered"
else
  print_status 1 "User registration failed"
fi

# 2. Login
echo "2. Logging in..."
LOGIN_RESPONSE=$(curl -s -X POST "$BASE_URL/login" \
  -H "Content-Type: application/json" \
  -d "{\"email\": \"$EMAIL\", \"password\": \"$PASSWORD\"}")

echo "Response: $LOGIN_RESPONSE"

if [[ $LOGIN_RESPONSE == *"success\":true"* ]]; then
  print_status 0 "User logged in"
else
  print_status 1 "User login failed"
fi

# 3. Forgot Password
echo "3. Requesting password reset..."
FORGOT_RESPONSE=$(curl -s -X POST "$BASE_URL/forgot-password" \
  -H "Content-Type: application/json" \
  -d "{\"email\": \"$EMAIL\"}")

echo "Response: $FORGOT_RESPONSE"

if [[ $FORGOT_RESPONSE == *"success\":true"* ]]; then
  print_status 0 "Password reset requested"
else
  print_status 1 "Password reset request failed"
fi

# Note: We can't easily test the full flow (verify email, reset password) 
# without extracting the tokens from the database or logs.
# But we can verify the endpoints respond correctly.

echo "Auth tests completed successfully!"