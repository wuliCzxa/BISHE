from OpenSSL import SSL, crypto
import socket

# 你的局域网IP
IP = "192.168.45.79"

# 生成密钥
key = crypto.PKey()
key.generate_key(crypto.TYPE_RSA, 2048)

# 生成证书
cert = crypto.X509()
cert.get_subject().C = "CN"
cert.get_subject().ST = "State"
cert.get_subject().L = "City"
cert.get_subject().O = "Organization"
cert.get_subject().OU = "OrgUnit"
cert.get_subject().CN = IP
cert.set_serial_number(1000)
cert.gmtime_adj_notBefore(0)
cert.gmtime_adj_notAfter(10 * 365 * 24 * 60 * 60)  # 10年有效期
cert.set_issuer(cert.get_subject())
cert.set_pubkey(key)
# cert.sign(key, "sha256")
# 关键修改：指定更安全的摘要算法
cert.sign(key, "sha256WithRSAEncryption") # 修改此处

# 保存文件
with open("cert.pem", "wb") as f:
    f.write(crypto.dump_certificate(crypto.FILETYPE_PEM, cert))
with open("key.pem", "wb") as f:
    f.write(crypto.dump_privatekey(crypto.FILETYPE_PEM, key))

print("✅ 证书生成成功！")
print("📄 cert.pem")
print("🔑 key.pem")