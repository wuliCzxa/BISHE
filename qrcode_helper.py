# """
# qrcode_helper.py
# 二维码生成辅助模块
# 用于生成动态二维码（微信扫描跳转小程序，浏览器扫描下载APK）
# """

# import io
# import qrcode
# from flask import send_file, request


# def is_wechat_browser():
#     """
#     检测是否为微信浏览器
#     通过User-Agent判断
#     """
#     user_agent = request.headers.get('User-Agent', '').lower()
#     return 'micromessenger' in user_agent


# def generate_qrcode_url(miniprogram_path, miniprogram_appid, apk_url):
#     """
#     根据访问来源生成不同的URL
    
#     Args:
#         miniprogram_path: 小程序页面路径
#         miniprogram_appid: 小程序AppID
#         apk_url: APK下载链接
    
#     Returns:
#         str: 要生成二维码的URL
#     """
#     if is_wechat_browser():
#         # 微信环境，返回小程序路径
#         # 方案1: 使用小程序URL Scheme（需要在微信公众平台配置）
#         # url = f"weixin://dl/business/?t={miniprogram_appid}"
        
#         # 方案2: 使用小程序码（推荐）
#         # 这里返回一个带参数的网页链接，网页会自动跳转到小程序
#         url = f"https://mp.weixin.qq.com/mp/waerrpage?appid={miniprogram_appid}&type=upgrade&upgradetype=3#wechat_redirect"
#     else:
#         # 非微信环境，返回APK下载链接
#         url = apk_url
    
#     return url


# def create_qrcode_image(data, box_size=10, border=2):
#     """
#     生成二维码图片
    
#     Args:
#         data: 要编码的数据（URL或文本）
#         box_size: 二维码单元格大小
#         border: 边框宽度
    
#     Returns:
#         BytesIO: 二维码图片的字节流
#     """
#     # 创建二维码对象
#     qr = qrcode.QRCode(
#         version=1,  # 控制二维码大小，1-40
#         error_correction=qrcode.constants.ERROR_CORRECT_H,  # 高容错率
#         box_size=box_size,
#         border=border,
#     )
    
#     # 添加数据
#     qr.add_data(data)
#     qr.make(fit=True)
    
#     # 生成图片
#     img = qr.make_image(fill_color="black", back_color="white")
    
#     # 保存到内存
#     img_io = io.BytesIO()
#     img.save(img_io, 'PNG')
#     img_io.seek(0)
    
#     return img_io


# def serve_qrcode(miniprogram_path, miniprogram_appid, apk_url):
#     """
#     生成并返回二维码图片
    
#     Args:
#         miniprogram_path: 小程序页面路径
#         miniprogram_appid: 小程序AppID
#         apk_url: APK下载链接
    
#     Returns:
#         Flask Response: 二维码图片响应
#     """
#     # 根据User-Agent生成不同的URL
#     url = generate_qrcode_url(miniprogram_path, miniprogram_appid, apk_url)
    
#     # 生成二维码图片
#     img_io = create_qrcode_image(url)
    
#     # 返回图片响应
#     return send_file(
#         img_io,
#         mimetype='image/png',
#         as_attachment=False,
#         download_name='qrcode.png'
#     )


# def get_qrcode_info():
#     """
#     获取二维码信息（用于前端显示提示）
    
#     Returns:
#         dict: 包含二维码类型和提示信息
#     """
#     if is_wechat_browser():
#         return {
#             "type": "wechat",
#             "title": "微信扫码",
#             "description": "打开微信小程序"
#         }
#     else:
#         return {
#             "type": "browser",
#             "title": "扫码下载",
#             "description": "下载移动应用APK"
#         }
"""
qrcode_helper.py
二维码生成辅助模块
用于生成动态二维码（微信扫描跳转小程序，浏览器扫描下载APK）
"""

import io
import qrcode
from flask import send_file, request, url_for


def is_wechat_browser():
    """
    检测是否为微信浏览器
    通过User-Agent判断
    """
    user_agent = request.headers.get('User-Agent', '').lower()
    return 'micromessenger' in user_agent


def generate_qrcode_url(miniprogram_path, miniprogram_appid, use_local_apk=True):
    """
    根据访问来源生成不同的URL
    
    Args:
        miniprogram_path: 小程序页面路径
        miniprogram_appid: 小程序AppID
        use_local_apk: 是否使用本地APK（True=使用本地路由，False=使用外部URL）
    
    Returns:
        str: 要生成二维码的URL
    """
    if is_wechat_browser():
        # 微信环境，返回小程序路径
        # 方案1: 使用小程序URL Scheme（需要在微信公众平台配置）
        # url = f"weixin://dl/business/?t={miniprogram_appid}"
        
        # 方案2: 使用小程序码（推荐）
        # 这里返回一个带参数的网页链接，网页会自动跳转到小程序
        url = f"https://mp.weixin.qq.com/mp/waerrpage?appid={miniprogram_appid}&type=upgrade&upgradetype=3#wechat_redirect"
    else:
        # 非微信环境，返回APK下载链接
        if use_local_apk:
            # 使用本地 APK 下载路由（推荐）
            # url_for 会根据当前请求自动生成完整URL（包含协议、域名、端口）
            url = url_for('download_apk', _external=True)
        else:
            # 使用外部 APK URL（如果 APK 托管在其他服务器）
            # 这种情况需要在 config.py 中配置 APK_DOWNLOAD_URL
            from flask import current_app
            url = current_app.config.get('APK_DOWNLOAD_URL', 
                                        'https://your-domain.com/downloads/app.apk')
    
    return url


def create_qrcode_image(data, box_size=10, border=2):
    """
    生成二维码图片
    
    Args:
        data: 要编码的数据（URL或文本）
        box_size: 二维码单元格大小
        border: 边框宽度
    
    Returns:
        BytesIO: 二维码图片的字节流
    """
    # 创建二维码对象
    qr = qrcode.QRCode(
        version=1,  # 控制二维码大小，1-40
        error_correction=qrcode.constants.ERROR_CORRECT_H,  # 高容错率
        box_size=box_size,
        border=border,
    )
    
    # 添加数据
    qr.add_data(data)
    qr.make(fit=True)
    
    # 生成图片
    img = qr.make_image(fill_color="black", back_color="white")
    
    # 保存到内存
    img_io = io.BytesIO()
    img.save(img_io, 'PNG')
    img_io.seek(0)
    
    return img_io


def serve_qrcode(miniprogram_path, miniprogram_appid, use_local_apk=True):
    """
    生成并返回二维码图片
    
    Args:
        miniprogram_path: 小程序页面路径
        miniprogram_appid: 小程序AppID
        use_local_apk: 是否使用本地APK下载（True=本地，False=使用APK_DOWNLOAD_URL）
    
    Returns:
        Flask Response: 二维码图片响应
    """
    # 根据User-Agent生成不同的URL
    url = generate_qrcode_url(miniprogram_path, miniprogram_appid, use_local_apk)
    
    # 生成二维码图片
    img_io = create_qrcode_image(url)
    
    # 返回图片响应
    return send_file(
        img_io,
        mimetype='image/png',
        as_attachment=False,
        download_name='qrcode.png'
    )


def get_qrcode_info():
    """
    获取二维码信息（用于前端显示提示）
    
    Returns:
        dict: 包含二维码类型和提示信息
    """
    if is_wechat_browser():
        return {
            "type": "wechat",
            "title": "微信扫码",
            "description": "打开微信小程序"
        }
    else:
        return {
            "type": "browser",
            "title": "扫码下载",
            "description": "下载移动应用APK"
        }
