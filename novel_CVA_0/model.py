import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as torch_init
torch.set_default_tensor_type('torch.FloatTensor')

def weight_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        #nn.init.kaiming_uniform_(m.weight, nonlinearity='leaky_relu')
        torch_init.xavier_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)
    # if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
    #     nn.init.xavier_uniform_(m.weight)
    #     if m.bias is not None:
    #         m.bias.data.fill_(0)


class CVA(nn.Module):
    def __init__(self, input_dim=1024):
        super(CVA, self).__init__()
        drop_out_rate = 0.1
        num_heads = 4
        self.cross_attention = nn.MultiheadAttention(embed_dim=input_dim, num_heads=num_heads, dropout=drop_out_rate, device='cuda')
        # self.self_attention = nn.MultiheadAttention(embed_dim=input_dim, num_heads=num_heads, dropout=drop_out_rate, device='cuda')
        # self.linear1 = nn.Linear(input_dim, 3072).cuda()
        # self.dropout = nn.Dropout(drop_out_rate).cuda()
        # self.linear2 = nn.Linear(3072, input_dim).cuda()
        # self.relu = nn.LeakyReLU(negative_slope=5e-2)

    def forward(self, feature1, feature2):
        """feature1: long path feature BXN1XC
           feature2: media path feature BXN2XC
        """


        #Layer_norm
        feature1 = F.layer_norm(feature1, [feature1.size(-1)]) # ln
        feature2 = F.layer_norm(feature2, [feature2.size(-1)])
        feature1 = feature1.permute(1, 0, 2) # N1 B C
        feature2 = feature2.permute(1, 0, 2)



        #cross attention
        out1, _ = self.cross_attention(query=feature1, key=feature2, value=feature2)  # N1 B C Test:32 1 1024
        out1 = out1 + feature1 #residual connection


        # layer norm
        # out1 = F.layer_norm(out1, out1.size())
        #
        # # self attention
        # out2, _ = self.self_attention(query=out1, key=out1, value=out1) #N1 B C
        # out2 = out2 + out1 #residual connection
        #
        # # layer norm
        # out2 = F.layer_norm(out2, out2.size())
        #
        # #mlp
        # out3 = self.linear1(out2)
        # out3 = self.relu(out3)
        # out3 = self.dropout(out3)
        # out3 = self.linear2(out3)
        # out3 = self.dropout(out3)
        #
        # out3 = out3 + out2 # N1 B C

        return out1 # B T C


class Aggregate(nn.Module):
    def __init__(self, len_feature):
        super(Aggregate, self).__init__()
        bn = nn.BatchNorm1d
        num_heads = 4
        self.len_feature = len_feature
        self.conv_1 = nn.Sequential(
            # nn.Conv1d(in_channels=len_feature, out_channels=512, kernel_size=3,
            #           stride=1,dilation=1, padding=1),
            nn.Conv1d(in_channels=len_feature, out_channels=512, kernel_size=3,
                      stride=1, dilation=1, padding=1),
            # nn.Conv1d(in_channels=len_feature, out_channels=512, kernel_size=3,
            #           stride=1, padding=1),
            nn.LeakyReLU(negative_slope=5e-2),
            bn(512)
            # nn.dropout(0.7)
        )
        self.conv_2 = nn.Sequential(
            # nn.Conv1d(in_channels=len_feature, out_channels=512, kernel_size=3,
            #           stride=1, dilation=2, padding=2),
            nn.Conv1d(in_channels=len_feature, out_channels=512, kernel_size=3,
                      stride=1, dilation=2, padding=2),
            # nn.Conv1d(in_channels=len_feature, out_channels=512, kernel_size=5,
            #           stride=1, padding=2),
            nn.LeakyReLU(negative_slope=5e-2),
            bn(512)
            # nn.dropout(0.7)
        )
        self.conv_3 = nn.Sequential(
            # nn.Conv1d(in_channels=len_feature, out_channels=512, kernel_size=3,
            #           stride=1, dilation=4, padding=4),
            nn.Conv1d(in_channels=len_feature, out_channels=512, kernel_size=3,
                      stride=1, dilation=4, padding=4),
            # nn.Conv1d(in_channels=len_feature, out_channels=512, kernel_size=9,
            #           stride=1, padding=4),
            nn.LeakyReLU(negative_slope=5e-2),
            bn(512)
            # nn.dropout(0.7),
        )
        self.conv_4 = nn.Sequential(
            nn.Conv1d(in_channels=len_feature*3, out_channels=512, kernel_size=1,
                      stride=1, padding=0, bias = False),
            nn.LeakyReLU(negative_slope=5e-2),
            # nn.dropout(0.7),
        )
        self.conv_5 = nn.Sequential(
            nn.Conv1d(in_channels=2048, out_channels=len_feature, kernel_size=3,
                      stride=1, padding=1, bias=False), # should we keep the bias?
            nn.LeakyReLU(negative_slope=5e-2),
            nn.BatchNorm1d(len_feature),
            # nn.dropout(0.7)
        )
        self.self_attention = nn.MultiheadAttention(embed_dim=512, num_heads=num_heads,
                                                    dropout=0.1, device='cuda')


    def forward(self, input1, input2, input3):
        # x: (t b c)
        x1 = input1.permute(1, 2, 0)  # B D T
        x2 = input2.permute(1, 2, 0)
        x3 = input3.permute(1, 2, 0)
        tensor_list = [x1, x2, x3]

        residual = torch.mean(torch.stack(tensor_list), dim=0)


        out1 = self.conv_1(x1)  # B D/2 T
        out2 = self.conv_2(x2)
        out3 = self.conv_3(x3)
        x = torch.cat([out1, out2, out3], dim=1)  # B 3D/2 T

        feature = torch.cat((x1, x2, x3), dim=1)
        out = self.conv_4(feature)
        out = out.permute(2, 0, 1) # T B D/2
        out = F.layer_norm(out, normalized_shape=[out.size(-1)])  # ln
        out, _ = self.self_attention(out, out, out) # T B D/2
        out = out.permute(1, 2, 0) #B D/2 T
        out = torch.cat((x, out), dim=1) # B 2D T
        out = self.conv_5(out)   # fuse all the features together
        out = out + residual
        out = out.permute(0, 2, 1)


        #out: (B, T, 1)

        return out

class Encoder(nn.Module):
    def __init__(self, input_dim=1024, seg_num=32):
        super(Encoder, self).__init__()
        num_heads = 8
        self.drop_out_rate = 0.1
        self.input_dim = input_dim
        self.min_temporal_dim = seg_num
        # self.self_attention = nn.MultiheadAttention(embed_dim=input_dim*3, num_heads=num_heads, dropout=self.drop_out_rate, device='cuda')
        # self.linear1 = nn.Linear(input_dim * 3, 3072)
        # self.dropout = nn.Dropout(self.drop_out_rate)
        # self.linear2 = nn.Linear(3072, input_dim)
        self.CVA1 = CVA(input_dim=input_dim)
        self.CVA2 = CVA(input_dim=input_dim)
        self.CVA3 = CVA(input_dim=input_dim)

        self.aggregate = Aggregate(len_feature=input_dim)

    def _align_temporal_dimension_across_views(self, feature):
        bs, time, c = feature.size()
        reshape_feature = feature.view(bs, self.min_temporal_dim, time // self.min_temporal_dim, c)

        return reshape_feature


    def _merge_along_channel(self, feature1, feature2, feature3):
        """
        feature1: [T N1 B C]
        feature2: [T N2 B C]
        feature3: [T N3 B C]
        """
        # feature1 = feature1.mean(1)
        # feature2 = feature2.mean(1)
        # feature3 = feature3.mean(1)

        feature = torch.cat((feature1, feature2, feature3), dim=-1) # T B 3C
        return feature


    # def _global_encoder(self, feature):
    #     """feature: T B 3C"""
    #     feature = self.dropout(feature)
    #     feature = F.layer_norm(feature, feature.size())
    #     att, _ = self.self_attention(query=feature, key=feature, value=feature) # T B 3C
    #     feature = feature + att
    #
    #
    #     feature = F.layer_norm(feature, feature.size())
    #     out1 = feature
    #
    #     out1 = self.linear1(out1)
    #     out1 = F.gelu(out1)
    #     out1 = self.dropout(out1)
    #     out1 = self.linear2(out1)
    #     out1 = self.dropout(out1)
    #
    #     feature = out1  # T B C
    #
    #
    #     return feature.permute(1, 0, 2) # B T C


    def forward(self, feature1, feature2, feature3):
        """
        feature1: [B T1 C]
        feature2: [B T2 C]
        feature3: [B T3 C]
        """

        # att1 = self.CVA1(feature1, feature2)  # T B C
        # att2 = self.CVA2(feature2, feature3)
        # att3 = self.CVA3(feature3, feature1)

        att1 = self.CVA1(feature2, feature1)  # T B C
        att2 = self.CVA2(feature3, feature2)
        att3 = self.CVA3(feature1, feature3)

        out1 = self.aggregate(att1, att2, att3) # B T C


        # out1 = self._merge_along_channel(att1, att2, att3) # T B 3C
        #
        # out2 = self._global_encoder(out1)


        return out1


class Model(nn.Module):
    def __init__(self, n_features, batch_size, seg_num=32):
        super(Model, self).__init__()
        self.batch_size = batch_size
        self.num_segments = seg_num
        self.k_abn = self.num_segments // 10
        self.k_nor = self.num_segments // 10

        self.Encoder = Encoder(input_dim=n_features, seg_num=seg_num)
        self.fc1 = nn.Linear(n_features, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 1)

        self.drop_out = nn.Dropout(0.2)
        self.relu = nn.LeakyReLU(negative_slope=5e-2)
        self.sigmoid = nn.Sigmoid()
        self.apply(weight_init)

    def forward(self, input1, input2, input3):
        """ input1 B T 1024"""
        k_abn = self.k_abn
        k_nor = self.k_nor
        ncrops = 1

        out = self.Encoder(input1, input2, input3)
        bs, t, f = out.size()
        features = self.drop_out(out) # B T D

        scores = self.relu(self.fc1(features))
        scores = self.drop_out(scores)
        scores = self.relu(self.fc2(scores))
        scores = self.drop_out(scores)
        scores = self.sigmoid(self.fc3(scores))
        scores = scores.view(bs, ncrops, -1).mean(1)
        scores = scores.unsqueeze(dim=2)
        # B * t * f
        normal_features = features[0:self.batch_size]
        normal_scores = scores[0:self.batch_size]

        abnormal_features = features[self.batch_size:]
        abnormal_scores = scores[self.batch_size:]

        feat_magnitudes = torch.norm(features, p=2, dim=2)
        feat_magnitudes = feat_magnitudes.view(bs, ncrops, -1).mean(1)
        nfea_magnitudes = feat_magnitudes[0:self.batch_size]  # normal feature magnitudes
        afea_magnitudes = feat_magnitudes[self.batch_size:]  # abnormal feature magnitudes
        n_size = nfea_magnitudes.shape[0]

        if nfea_magnitudes.shape[0] == 1:  # this is for inference, the batch size is 1
            afea_magnitudes = nfea_magnitudes
            abnormal_scores = normal_scores
            abnormal_features = normal_features

        select_idx = torch.ones_like(nfea_magnitudes)
        select_idx = self.drop_out(select_idx)

        #######  process abnormal videos -> select top3 feature magnitude  #######
        afea_magnitudes_drop = afea_magnitudes * select_idx
        idx_abn = torch.topk(afea_magnitudes_drop, k_abn, dim=1)[1]
        idx_abn_feat = idx_abn.unsqueeze(2).expand([-1, -1, abnormal_features.shape[2]])

        abnormal_features = abnormal_features.view(n_size, ncrops, t, f) # B X N X T X F
        abnormal_features = abnormal_features.permute(1, 0, 2, 3)  # N X B X T X F

        total_select_abn_feature = torch.zeros(0, device=input1.device)
        for abnormal_feature in abnormal_features:
            feat_select_abn = torch.gather(abnormal_feature, 1, idx_abn_feat)   # top 3 features magnitude in abnormal bag
            total_select_abn_feature = torch.cat((total_select_abn_feature, feat_select_abn))

        idx_abn_score = idx_abn.unsqueeze(2).expand([-1, -1, abnormal_scores.shape[2]])
        score_abnormal = torch.mean(torch.gather(abnormal_scores, 1, idx_abn_score), dim=1)  # top 3 scores in abnormal bag based on the top-3 magnitude


        ####### process normal videos -> select top3 feature magnitude #######

        select_idx_normal = torch.ones_like(nfea_magnitudes)
        select_idx_normal = self.drop_out(select_idx_normal)
        nfea_magnitudes_drop = nfea_magnitudes * select_idx_normal
        idx_normal = torch.topk(nfea_magnitudes_drop, k_nor, dim=1)[1]
        idx_normal_feat = idx_normal.unsqueeze(2).expand([-1, -1, normal_features.shape[2]])

        normal_features = normal_features.view(n_size, ncrops, t, f)
        normal_features = normal_features.permute(1, 0, 2, 3) # 1 B T D

        total_select_nor_feature = torch.zeros(0, device=input1.device)
        for nor_fea in normal_features:
            feat_select_normal = torch.gather(nor_fea, 1, idx_normal_feat)  # top 3 features magnitude in normal bag (hard negative)
            total_select_nor_feature = torch.cat((total_select_nor_feature, feat_select_normal))

        idx_normal_score = idx_normal.unsqueeze(2).expand([-1, -1, normal_scores.shape[2]])
        score_normal = torch.mean(torch.gather(normal_scores, 1, idx_normal_score), dim=1) # top 3 scores in normal bag

        feat_select_abn = total_select_abn_feature
        feat_select_normal = total_select_nor_feature

        return score_abnormal, score_normal, feat_select_abn, feat_select_normal, feat_select_abn, feat_select_abn, scores, feat_select_abn, feat_select_abn, feat_magnitudes