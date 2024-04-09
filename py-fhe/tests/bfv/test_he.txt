import os
import seal
#from seal import EncryptionParameters, scheme_type, SEALContext, KeyGenerator, Encryptor, Decryptor, IntegerEncoder, Plaintext, Ciphertext

def encrypt_file(file_path, public_key):
    # Read the file content
    with open(file_path, 'rb') as file:
        data = file.read()

    # Initialize SEAL parameters
    parms = EncryptionParameters(scheme_type.BFV)
    poly_modulus_degree = 4096
    parms.set_poly_modulus_degree(poly_modulus_degree)
    parms.set_coeff_modulus(EncryptionParameters().coeff_modulus_128(poly_modulus_degree))
    parms.set_plain_modulus(256)
    
    # Create SEAL context
    context = SEALContext(parms)
    
    # Create an encryptor
    encryptor = Encryptor(context, public_key)
    
    # Encrypt the data
    data_plaintext = Plaintext()
    data_plaintext.set_data(data)
    data_ciphertext = Ciphertext()
    encryptor.encrypt(data_plaintext, data_ciphertext)
    
    return data_ciphertext

def decrypt_file(ciphertext, secret_key):
    # Create a decryptor
    decryptor = Decryptor(SEALContext(ciphertext.parms()), secret_key)
    
    # Decrypt the ciphertext
    decrypted_data = Plaintext()
    decryptor.decrypt(ciphertext, decrypted_data)
    
    return decrypted_data.to_string()

if __name__ == "__main__":
    # Generate keys
    parms = EncryptionParameters(scheme_type.BFV)
    poly_modulus_degree = 4096
    parms.set_poly_modulus_degree(poly_modulus_degree)
    parms.set_coeff_modulus(EncryptionParameters().coeff_modulus_128(poly_modulus_degree))
    parms.set_plain_modulus(256)
    context = SEALContext(parms)
    keygen = KeyGenerator(context)
    public_key = keygen.public_key()
    secret_key = keygen.secret_key()
    
    # Encrypt the file
    file_path = "BC-TCGA-Normal.txt"
    encrypted_data = encrypt_file(file_path, public_key)
    
    # Decrypt the file
    decrypted_data = decrypt_file(encrypted_data, secret_key)
    
    # Write the decrypted data to a new file
    with open("decrypted_" + file_path, 'wb') as file:
        file.write(decrypted_data)
