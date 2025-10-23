import pyotp as tp
totp_key = 'BZ6WLM363WLWA744RZIQ2WZCUPH2F4UD'
t = tp.TOTP(totp_key).now()
print(t)