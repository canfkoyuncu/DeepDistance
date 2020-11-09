% goldMap  : The map for the cell annotations. Pixels of each cell  
%            should be marked with a positive number. Background pixels   
%            should be marked with zero.
% alpha    : The decay ratio in the inner distance calculation. Its
%            default value is set to 0.1.
%
function [innerDistMap, outerDistMap] = calculateDistances(goldMap, alpha)

if (nargin == 1)
    alpha = 0.1;
end

[centxy, bndbox] = calculateCentroid(goldMap);
centxy = round(centxy);

innerDistMap = findInnerDistance(goldMap, centxy, alpha);
outerDistMap = findOuterDistance(goldMap, bndbox);

end
%------------------------------------------------------------------%
%------------------------------------------------------------------%
%------------------------------------------------------------------%
function outerDistMap = findOuterDistance(gs, bndbox)

gsno = size(bndbox, 1);
[dx, dy] = size(gs);

outerDistMap = zeros(dx, dy);
for gid = 1 : gsno
    bndPix = findBoundaryPixels(gs, gid, bndbox);
    currMaxD = 0;
    for i = bndbox(gid, 1) : bndbox(gid, 2)
        for j = bndbox(gid, 3) : bndbox(gid, 4)
            if gs(i, j) == gid
                minD = findMinDistance(bndPix, i, j);
                if minD > currMaxD
                    currMaxD = minD;
                end
                outerDistMap(i, j) = minD;
            end
        end
    end
    for i = bndbox(gid, 1) : bndbox(gid, 2)
        for j = bndbox(gid, 3) : bndbox(gid, 4)
            if gs(i, j) == gid
                outerDistMap(i, j) = outerDistMap(i, j) / currMaxD;
            end
        end
    end
end

end
%------------------------------------------------------------------%
%------------------------------------------------------------------%
%------------------------------------------------------------------%
% Finds the boundary on the background pixels
function bndPix = findBoundaryPixels(gs, gid, bndbox)

minx = bndbox(gid, 1);
maxx = bndbox(gid, 2);
miny = bndbox(gid, 3);
maxy = bndbox(gid, 4);
[dx, dy] = size(gs);
bndPix = zeros((maxx - minx + 1) * (maxy - miny + 1), 2);
bndNo = 0;
for i = minx : maxx
    for j = miny : maxy
        if gs(i, j) == gid
            if  i ~= 1 && gs(i - 1, j) ~= gid
                bndNo = bndNo + 1;
                bndPix(bndNo, 1) = i - 1;
                bndPix(bndNo, 2) = j;
            end
            
            if  i ~= dx && gs(i + 1, j) ~= gid
                bndNo = bndNo + 1;
                bndPix(bndNo, 1) = i + 1;
                bndPix(bndNo, 2) = j;
            end
            
            if  j ~= 1 && gs(i, j - 1) ~= gid
                bndNo = bndNo + 1;
                bndPix(bndNo, 1) = i;
                bndPix(bndNo, 2) = j - 1;
            end
            
            if  j ~= dy && gs(i, j + 1) ~= gid
                bndNo = bndNo + 1;
                bndPix(bndNo, 1) = i;
                bndPix(bndNo, 2) = j + 1;
            end
        end
    end
end
bndPix = bndPix(1 : bndNo, :);

end
%------------------------------------------------------------------%
%------------------------------------------------------------------%
%------------------------------------------------------------------%
function innerDistMap = findInnerDistance(gs, centxy, alpha)

[dx, dy] = size(gs);
innerDistMap = zeros(dx, dy);
for i = 1 : dx
    for j = 1 : dy
        if gs(i, j)
            minD = findMinDistance(centxy, i, j);
            innerDistMap(i, j) = 1 / (1 + alpha * minD);
        end
    end
end

end
%------------------------------------------------------------------%
%------------------------------------------------------------------%
%------------------------------------------------------------------%
function minD = findMinDistance(candxy, cx, cy)

no = size(candxy, 1);
minD = (candxy(1, 1) - cx)^2 + (candxy(1, 2) - cy)^2;
for i = 2 : no
    d = (candxy(i, 1) - cx)^2 + (candxy(i, 2) - cy)^2;
    if minD > d
        minD = d;
    end
end
minD = sqrt(minD);

end
%------------------------------------------------------------------%
%------------------------------------------------------------------%
%------------------------------------------------------------------%
function [centxy, bndbox] = calculateCentroid(gs)

centno = max(max(gs));
centxy = zeros(centno, 2);
areas = zeros(centno, 1);
[dx, dy] = size(gs);
bndbox = zeros(centno, 4);
bndbox(:, 1) = dx + 1;
bndbox(:, 3) = dy + 1;
for i = 1 : dx
    for j = 1 : dy
        if gs(i, j)
            gid = gs(i, j);
            areas(gid) = areas(gid) + 1;
            centxy(gid, 1) = centxy(gid, 1) + i;
            centxy(gid, 2) = centxy(gid, 2) + j;
            
            if i < bndbox(gid, 1), bndbox(gid, 1) = i;  end
            if i > bndbox(gid, 2), bndbox(gid, 2) = i;  end
            if j < bndbox(gid, 3), bndbox(gid, 3) = j;  end
            if j > bndbox(gid, 4), bndbox(gid, 4) = j;  end
        end
    end
end
for i = 1 : centno
    centxy(i, 1) = centxy(i, 1) / areas(i);
    centxy(i, 2) = centxy(i, 2) / areas(i);
end

end
%------------------------------------------------------------------%
%------------------------------------------------------------------%
%------------------------------------------------------------------%
