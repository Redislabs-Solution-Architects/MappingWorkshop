## General design
# Per User
1. registered user is added to Redis in the "users" (SET type) key
2. user has a **user:<user_id>:elements** (SET type) with the all registered elements that it tracks
3. user has a **user:<user_id>:stream** (STREAM type) with the elements that belong to the user_id that were updated
4. from the user_id stream, it is push to the frontend with socket.io

### Instructions

1. Fill the functions that has comment at their top with ## Implement:
2. the comment ontop of each function has explanation on what to fill
