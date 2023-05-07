use image::{DynamicImage, GenericImageView, ImageBuffer, Rgb, Rgba};
use nalgebra::Vector2;
use std::path::Path;
use imageproc::{drawing::{draw_cross_mut, Canvas}, point::Point};

use vslam_core::keyframe::Feature;
use vslam_frontend::sift::SIFT;

fn main() {
    // 读取两幅图像
    let image = image::open(Path::new("C:/Users/Administrator/Documents/vslam/data/V1_01_easy/mav0/cam0/data/1403715389562142976.png")).unwrap();
    //let image2 = image::open(Path::new("C:/Users/Administrator/Documents/vslam/data/V1_01_easy/mav0/cam0/data/1403715413312143104.png")).unwrap();

    //
    let features = SIFT::extract_features(&image);

    // 创建一个可变图像以绘制特征点
    let mut output_image = DynamicImage::ImageRgb8(image.to_rgb8());

    // 在图像上绘制特征点
    for feature in features.iter() {
        let point=feature.location;
        let color:Rgba<u8> = Rgba([0, 255, 0,255]);
        draw_cross_mut(&mut output_image, color, point.x as i32,point.y as i32);
    }

    // 显示带有特征点的图像
    output_image.save("output_image.jpg").unwrap();


}
