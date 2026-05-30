//package com.mycompany.flaskapp;
//
//import com.getcapacitor.BridgeActivity;
//
//public class MainActivity extends BridgeActivity {}
//package com.mycompany.flaskapp;
//
//import com.getcapacitor.BridgeActivity;
//import android.os.Bundle;
//import android.webkit.PermissionRequest;
//import android.webkit.WebChromeClient;
//import android.webkit.WebView;
//import android.webkit.SslErrorHandler;
//import android.webkit.WebViewClient;
//import android.net.http.SslError;
//
//import android.Manifest;
//import android.content.pm.PackageManager;
//import android.os.Build;
//import androidx.core.app.ActivityCompat;
//import androidx.core.content.ContextCompat;
//
//import java.util.ArrayList;
//import java.util.List;
//
//public class MainActivity extends BridgeActivity {
//
//    private static final int PERMISSION_REQUEST_CODE = 100;
//
//    @Override
//    protected void onCreate(Bundle savedInstanceState) {
//        super.onCreate(savedInstanceState);
//
//        WebView webView = getBridge().getWebView();
//
//        // 开启硬件加速
//        webView.setLayerType(WebView.LAYER_TYPE_HARDWARE, null);
//
//        // 忽略 SSL 自签名证书错误（HTTPS 必须）
//        webView.setWebViewClient(new WebViewClient() {
//            @Override
//            public void onReceivedSslError(WebView view, SslErrorHandler handler, SslError error) {
//                handler.proceed();
//            }
//        });
//
//        // 授权网页摄像头 / 麦克风权限（核心！）
//        webView.setWebChromeClient(new WebChromeClient() {
//            @Override
//            public void onPermissionRequest(PermissionRequest request) {
//                request.grant(request.getResources());
//            }
//        });
//
//        // 动态申请权限（根据 Android 版本区分）
//        requestNecessaryPermissions();
//    }
//
//    /**
//     * 根据 Android 版本动态申请必要的权限
//     */
//    private void requestNecessaryPermissions() {
//        List<String> permissionsToRequest = new ArrayList<>();
//
//        // 摄像头和麦克风权限（所有版本都需要）
//        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
//                != PackageManager.PERMISSION_GRANTED) {
//            permissionsToRequest.add(Manifest.permission.CAMERA);
//        }
//        if (ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO)
//                != PackageManager.PERMISSION_GRANTED) {
//            permissionsToRequest.add(Manifest.permission.RECORD_AUDIO);
//        }
//
//        // Android 13+ (API 33+)：使用细粒度媒体权限
//        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
//            // 图片权限
//            if (ContextCompat.checkSelfPermission(this, Manifest.permission.READ_MEDIA_IMAGES)
//                    != PackageManager.PERMISSION_GRANTED) {
//                permissionsToRequest.add(Manifest.permission.READ_MEDIA_IMAGES);
//            }
//            // 视频权限
//            if (ContextCompat.checkSelfPermission(this, Manifest.permission.READ_MEDIA_VIDEO)
//                    != PackageManager.PERMISSION_GRANTED) {
//                permissionsToRequest.add(Manifest.permission.READ_MEDIA_VIDEO);
//            }
//            // 音频权限（如果需要）
//            if (ContextCompat.checkSelfPermission(this, Manifest.permission.READ_MEDIA_AUDIO)
//                    != PackageManager.PERMISSION_GRANTED) {
//                permissionsToRequest.add(Manifest.permission.READ_MEDIA_AUDIO);
//            }
//        }
//        // Android 6 - 12 (API 23-32)：使用传统存储权限
//        else if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
//            if (ContextCompat.checkSelfPermission(this, Manifest.permission.READ_EXTERNAL_STORAGE)
//                    != PackageManager.PERMISSION_GRANTED) {
//                permissionsToRequest.add(Manifest.permission.READ_EXTERNAL_STORAGE);
//            }
//            if (ContextCompat.checkSelfPermission(this, Manifest.permission.WRITE_EXTERNAL_STORAGE)
//                    != PackageManager.PERMISSION_GRANTED) {
//                permissionsToRequest.add(Manifest.permission.WRITE_EXTERNAL_STORAGE);
//            }
//        }
//
//        // 如果有需要申请的权限，一次性申请
//        if (!permissionsToRequest.isEmpty()) {
//            ActivityCompat.requestPermissions(
//                    this,
//                    permissionsToRequest.toArray(new String[0]),
//                    PERMISSION_REQUEST_CODE
//            );
//        }
//    }
//
//    /**
//     * 处理权限申请结果
//     */
//    @Override
//    public void onRequestPermissionsResult(int requestCode, String[] permissions, int[] grantResults) {
//        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
//
//        if (requestCode == PERMISSION_REQUEST_CODE) {
//            // 检查哪些权限被授予或拒绝
//            for (int i = 0; i < permissions.length; i++) {
//                if (grantResults[i] == PackageManager.PERMISSION_GRANTED) {
//                    // 权限已授予
//                    android.util.Log.d("MainActivity", "权限已授予: " + permissions[i]);
//                } else {
//                    // 权限被拒绝
//                    android.util.Log.w("MainActivity", "权限被拒绝: " + permissions[i]);
//
//                    // 可选：如果用户永久拒绝，引导到设置页面
//                    if (!ActivityCompat.shouldShowRequestPermissionRationale(this, permissions[i])) {
//                        android.util.Log.w("MainActivity", "用户永久拒绝了权限，可引导到设置页面");
//                        // 可以显示一个对话框，引导用户到设置中手动开启权限
//                    }
//                }
//            }
//        }
//    }
//}

package com.mycompany.flaskapp;

import com.getcapacitor.BridgeActivity;
import android.os.Bundle;
import android.webkit.PermissionRequest;
import android.webkit.WebChromeClient;
import android.webkit.WebView;
import android.webkit.SslErrorHandler;
import android.webkit.WebViewClient;
import android.webkit.ValueCallback;
import android.net.http.SslError;
import android.net.Uri;

import android.Manifest;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.os.Build;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;
import androidx.annotation.Nullable;

import java.util.ArrayList;
import java.util.List;

public class MainActivity extends BridgeActivity {

    private static final int PERMISSION_REQUEST_CODE = 100;
    private static final int FILE_CHOOSER_REQUEST_CODE = 200;

    // 文件选择器回调
    private ValueCallback<Uri[]> mFilePathCallback;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        WebView webView = getBridge().getWebView();

        // 开启硬件加速
        webView.setLayerType(WebView.LAYER_TYPE_HARDWARE, null);

        // 忽略 SSL 自签名证书错误（HTTPS 必须）
        webView.setWebViewClient(new WebViewClient() {
            @Override
            public void onReceivedSslError(WebView view, SslErrorHandler handler, SslError error) {
                handler.proceed();
            }
        });

        // 配置 WebChromeClient 支持文件上传和权限
        webView.setWebChromeClient(new WebChromeClient() {
            // 授权网页摄像头 / 麦克风权限
            @Override
            public void onPermissionRequest(PermissionRequest request) {
                request.grant(request.getResources());
            }

            // ⚠️ 关键：处理文件选择（Android 5.0+）
            @Override
            public boolean onShowFileChooser(
                    WebView webView,
                    ValueCallback<Uri[]> filePathCallback,
                    FileChooserParams fileChooserParams) {

                // 如果有旧的回调，先取消
                if (mFilePathCallback != null) {
                    mFilePathCallback.onReceiveValue(null);
                }

                mFilePathCallback = filePathCallback;

                // 创建文件选择器 Intent
                Intent intent = fileChooserParams.createIntent();

                // 支持多选（如果网页允许）
                if (fileChooserParams.getMode() == FileChooserParams.MODE_OPEN_MULTIPLE) {
                    intent.putExtra(Intent.EXTRA_ALLOW_MULTIPLE, true);
                }

                try {
                    startActivityForResult(intent, FILE_CHOOSER_REQUEST_CODE);
                    return true;
                } catch (Exception e) {
                    mFilePathCallback = null;
                    android.util.Log.e("MainActivity", "文件选择器启动失败", e);
                    return false;
                }
            }
        });

        // 动态申请权限（根据 Android 版本区分）
        requestNecessaryPermissions();
    }

    /**
     * 根据 Android 版本动态申请必要的权限
     */
    private void requestNecessaryPermissions() {
        List<String> permissionsToRequest = new ArrayList<>();

        // 摄像头和麦克风权限（所有版本都需要）
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
                != PackageManager.PERMISSION_GRANTED) {
            permissionsToRequest.add(Manifest.permission.CAMERA);
        }
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO)
                != PackageManager.PERMISSION_GRANTED) {
            permissionsToRequest.add(Manifest.permission.RECORD_AUDIO);
        }

        // Android 13+ (API 33+)：使用细粒度媒体权限
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            // 图片权限
            if (ContextCompat.checkSelfPermission(this, Manifest.permission.READ_MEDIA_IMAGES)
                    != PackageManager.PERMISSION_GRANTED) {
                permissionsToRequest.add(Manifest.permission.READ_MEDIA_IMAGES);
            }
            // 视频权限
            if (ContextCompat.checkSelfPermission(this, Manifest.permission.READ_MEDIA_VIDEO)
                    != PackageManager.PERMISSION_GRANTED) {
                permissionsToRequest.add(Manifest.permission.READ_MEDIA_VIDEO);
            }
            // 音频权限（如果需要）
            if (ContextCompat.checkSelfPermission(this, Manifest.permission.READ_MEDIA_AUDIO)
                    != PackageManager.PERMISSION_GRANTED) {
                permissionsToRequest.add(Manifest.permission.READ_MEDIA_AUDIO);
            }
        }
        // Android 6 - 12 (API 23-32)：使用传统存储权限
        else if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
            if (ContextCompat.checkSelfPermission(this, Manifest.permission.READ_EXTERNAL_STORAGE)
                    != PackageManager.PERMISSION_GRANTED) {
                permissionsToRequest.add(Manifest.permission.READ_EXTERNAL_STORAGE);
            }
            if (ContextCompat.checkSelfPermission(this, Manifest.permission.WRITE_EXTERNAL_STORAGE)
                    != PackageManager.PERMISSION_GRANTED) {
                permissionsToRequest.add(Manifest.permission.WRITE_EXTERNAL_STORAGE);
            }
        }

        // 如果有需要申请的权限，一次性申请
        if (!permissionsToRequest.isEmpty()) {
            ActivityCompat.requestPermissions(
                    this,
                    permissionsToRequest.toArray(new String[0]),
                    PERMISSION_REQUEST_CODE
            );
        }
    }

    /**
     * 处理权限申请结果
     */
    @Override
    public void onRequestPermissionsResult(int requestCode, String[] permissions, int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);

        if (requestCode == PERMISSION_REQUEST_CODE) {
            // 检查哪些权限被授予或拒绝
            for (int i = 0; i < permissions.length; i++) {
                if (grantResults[i] == PackageManager.PERMISSION_GRANTED) {
                    // 权限已授予
                    android.util.Log.d("MainActivity", "权限已授予: " + permissions[i]);
                } else {
                    // 权限被拒绝
                    android.util.Log.w("MainActivity", "权限被拒绝: " + permissions[i]);

                    // 可选：如果用户永久拒绝，引导到设置页面
                    if (!ActivityCompat.shouldShowRequestPermissionRationale(this, permissions[i])) {
                        android.util.Log.w("MainActivity", "用户永久拒绝了权限，可引导到设置页面");
                    }
                }
            }
        }
    }

    /**
     * 处理文件选择器返回的结果（关键！）
     */
    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        super.onActivityResult(requestCode, resultCode, data);

        if (requestCode == FILE_CHOOSER_REQUEST_CODE) {
            if (mFilePathCallback == null) {
                return;
            }

            Uri[] results = null;

            // 用户选择了文件
            if (resultCode == RESULT_OK && data != null) {
                String dataString = data.getDataString();

                // 处理单个文件
                if (dataString != null) {
                    results = new Uri[]{Uri.parse(dataString)};
                }
                // 处理多个文件
                else if (data.getClipData() != null) {
                    int count = data.getClipData().getItemCount();
                    results = new Uri[count];
                    for (int i = 0; i < count; i++) {
                        results[i] = data.getClipData().getItemAt(i).getUri();
                    }
                }
            }

            // 将结果传回 WebView
            mFilePathCallback.onReceiveValue(results);
            mFilePathCallback = null;
        }
    }
}