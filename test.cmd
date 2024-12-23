# pbrt --outfile .\outfile\villa-daylight-ris.exr "D:\RayTracing\pbrt-v4\scenes\villa\villa-daylight-ris.pbrt" --display-server localhost:14158

# pbrt --outfile .\outfile\villa-daylight-test.exr "D:\RayTracing\pbrt-v4\scenes\villa\villa-daylight-test.pbrt" --display-server localhost:14158 --gpu --gpu-device 0
# pbrt --outfile .\outfile\villa-daylight.exr "D:\RayTracing\pbrt-v4\scenes\villa\villa-daylight.pbrt" --display-server localhost:14158 --gpu --gpu-device 0 

# pbrt --outfile .\outfile\villa-daylight.exr "D:\RayTracing\pbrt-v4\scenes\villa\villa-daylight.pbrt" --display-server localhost:14158

# pbrt --outfile "D:\RayTracing\pbrt-v4\images\villa-daylight-ris.exr" "D:\RayTracing\pbrt-v4\scenes\villa\villa-daylight-ris.pbrt" --display-server localhost:14158

pbrt --outfile "D:\RayTracing\pbrt-v4\images\bistro_cafe_ris.exr" "D:\RayTracing\pbrt-v4\scenes\bistro\bistro_cafe_ris.pbrt" --display-server localhost:14158

pbrt --outfile "D:\RayTracing\pbrt-v4\images\bistro_cafe_path.exr" "D:\RayTracing\pbrt-v4\scenes\bistro\bistro_cafe.pbrt" --display-server localhost:14158

pbrt --outfile "D:\RayTracing\pbrt-v4\images\bistro_cafe_gt.exr" "D:\RayTracing\pbrt-v4\scenes\bistro\bistro_cafe_gt.pbrt" --display-server localhost:14158 --gpu --gpu-device 0

pbrt --outfile "D:\RayTracing\pbrt-v4\images\zero_day_path.exr" "D:\RayTracing\pbrt-v4\scenes\zero-day\frame35_path.pbrt" --display-server localhost:14158

pbrt --outfile "D:/RayTracing/pbrt-v4/images/bistro_path_v2.exr" "D:/RayTracing/Scenes/Bistro_v5_2_fbx_png/bistro_path.pbrt" --display-server localhost:14158

pbrt --outfile "D:\RayTracing\pbrt-v4\images\bistro.exr" "D:\RayTracing\Scenes\Bistro_v5_2_fbx_png\bistro_path.pbrt" --interactive

--mse-reference-image "D:/RayTracing/pbrt-v4/images/bistro_path_48000.exr" --mse-reference-out "D:/RayTracing/pbrt-v4/images/bistro_path_mse.txt"