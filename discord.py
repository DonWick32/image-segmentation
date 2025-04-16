import requests

webhook_url = "https://discord.com/api/webhooks/1360661259921526906/8QFXS5KTz_Z4viuHFmYRlnqfryroqPse_K-Be1mkeEUarWYvqhx6baJpayJI6uPlFijs"

thala_url = "https://discord.com/api/webhooks/1360663640046571580/mNocZ3tLWiUVaMQTnOWqVRJU-HdI9onQuw0Wcr1xn8ZxRdvY51kuf9IcZ2qxRIBE21-x"

requests.post(thala_url, { "content": "Hello world" , "username" : "gaejul" })