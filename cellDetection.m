% estimatedInner  : The inner map estimated by the multi-task network.
% h               : The h value used in the h-maxima transform which
%                   suppresses the maxima whose height is less than h.
%
% cellCenters     : The map of estimated cell locations. Pixels of each 
%                   cell location are marked with a positive number.
%                   Other pixels are marked with zero.
%
function cellCenters = cellDetection(estimatedInner, h)

hmap = imhmax(estimatedInner, h);
cellCenters = imregionalmax(hmap);
cellCenters = bwlabeln(cellCenters, 4);

end
