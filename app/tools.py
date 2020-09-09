import logging
import os
from cryptography.fernet import Fernet


def save_encrypted(password, key, filename):
    """
    Encrypts a password and saves it to file
    """
    encrypted_password = Fernet(key).encrypt(password.encode())
    with open(filename, "wb") as f:
        f.write(encrypted_password)


def decrypt_password(encrypted_password, key):
    """
    Decrypts an encrypted password
    """
    decrypted_password = Fernet(key).decrypt(encrypted_password)
    return decrypted_password.decode()


def send_email(sender='your_name@bk.ru',
               receivers=['myname@gmail.com'],
               subject='terminal notification',
               body=''):
    import smtplib
    from os.path import expanduser

    filename = '/usr/share/email.pass'
    if os.path.isfile(filename):
        # key = Fernet.generate_key()
        key = b'FMfaBOYAoB4I7DE4VCqLEJQMCsDB65VjogOkIvhbDsY='
        with open(filename, 'rb') as f:
            password = decrypt_password(f.read(), key)
    else:
        logging.error('file %s does not exist' % filename)
        return

    msg = "\r\n".join([
        "From: %s" % sender,
        "To: %s" % (','.join(receivers)),
        "Subject: %s" % subject,
        "%s" % body
    ])

    server = smtplib.SMTP('smtp.mail.ru:587')
    server.ehlo()
    server.starttls()

    server.login('rghg@bk.ru', password)
    server.sendmail(sender, receivers, msg)
    server.quit()
