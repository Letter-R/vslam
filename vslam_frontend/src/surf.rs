use image::{DynamicImage,GrayImage};
use nalgebra::Vector2;
use vslam_core::keyframe::Feature;


pub struct SURF{
    pub location:Vector2<f64>,
    pub descriptor:[u64;2],
}
impl Feature for SURF {
    fn extract_features(image:&DynamicImage)->Vec<Self> where Self:Sized {
        let gray_image = image::imageops::grayscale(image);
        // 计算积分图
        let integral_image = compute_integral_image(&gray_image);
        
        
        todo!()
    }

    fn match_features(feature1:&Vec<Self>,feature2:&Vec<Self>)->Vec<(usize,usize)> where Self:Sized {
        todo!()
    }
}

/// 计算积分图，与输入图像大小相等
fn compute_integral_image(image: &image::GrayImage) -> Vec<Vec<f64>> {
    let (width, height) = image.dimensions();
    let mut integral_image = vec![vec![0.0; height as usize]; width as usize];

    for y in 0..height {
        for x in 0..width {
            let mut sum = image.get_pixel(x, y)[0] as f64;

            if y > 0 {
                sum += integral_image[x as usize][(y - 1) as usize];
            }
            if x > 0 {
                sum += integral_image[(x - 1) as usize][y as usize];
            }
            if x > 0 && y > 0 {
                sum -= integral_image[(x - 1) as usize][(y - 1) as usize];
            }

            integral_image[x as usize][y as usize] = sum;
        }
    }

    integral_image
}

/// 计算给定尺度的盒子滤波器响应
fn box_filter_response(integral_image: &Vec<Vec<f64>>, x: u32, y: u32, scale: u32) -> f64 {
    let x = x as i32;
    let y = y as i32;
    let half_scale = (scale / 2) as i32;
    let quarter_scale = (scale / 4) as i32;

    let dxx = dxx(integral_image, x, y, half_scale);
    let dyy = dyy(integral_image, x, y, half_scale);
    let dxy = dxy(integral_image, x, y, quarter_scale);

    dxx * dyy - dxy * dxy
}

/// 计算给定区域的像素和
fn box_filter(integral_image: &Vec<Vec<f64>>, x: i32, y: i32, width: i32, height: i32) -> f64 {
    let x1 = (x - width / 2).max(0) as usize;// 左
    let x2 = (x + width / 2).min(integral_image.len() as i32 - 1) as usize;// 右
    let y1 = (y - height / 2).max(0) as usize;// 上
    let y2 = (y + height / 2).min(integral_image[0].len() as i32 - 1) as usize;// 下

    let a = integral_image[x1][y1];// 左上
    let b = integral_image[x2][y1];// 右上
    let c = integral_image[x1][y2];// 左下
    let d = integral_image[x2][y2];// 右下

    d - b - c + a
}

/// 近似高斯二阶微分模板Dxx
fn dxx(integral_image: &Vec<Vec<f64>>, x: i32, y: i32, scale: i32) -> f64 {
    box_filter(integral_image, x - scale, y, scale, 2 * scale) -
    2.0 * box_filter(integral_image, x, y, scale, 2 * scale) +
    box_filter(integral_image, x + scale, y, scale, 2 * scale)
}

fn dyy(integral_image: &Vec<Vec<f64>>, x: i32, y: i32, scale: i32) -> f64 {
    box_filter(integral_image, x, y - scale, 2 * scale, scale) -
    2.0 * box_filter(integral_image, x, y, 2 * scale, scale) +
    box_filter(integral_image, x, y + scale, 2 * scale, scale)
}


fn dxy(integral_image: &Vec<Vec<f64>>, x: i32, y: i32, scale: i32) -> f64 {
    box_filter(integral_image, x - scale, y - scale, scale, scale) +
    box_filter(integral_image, x, y, scale, scale) -
    box_filter(integral_image, x, y - scale, scale, scale) -
    box_filter(integral_image, x - scale, y, scale, scale)
}
